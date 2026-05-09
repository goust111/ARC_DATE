# Variant of 271d71e2
# Dimension: D8 — 变换类型 (transform)
# Change: recolor — 灰色(5)变为橙色(7)，容器整体平移，目标线保持不动... → 容器到达后与目标线合并
#
# Design analysis
# ===============
# Reference-task note:
# - The prompt points to source task `7c66cb00` for docking inspiration.
# - I searched the local workspace for a solver file keyed by that task id and did
#   not find one, so the implementation below uses the variant brief itself as the
#   docking baseline and keeps all non-D8 structure from the original 271d71e2.
#
# a) Concrete behavior and trace
# - Input objects stay the same as the original task: magenta background, black
#   rectangular containers, grey/orange interior, maroon head line adjacent to one
#   border edge, matching maroon destination line farther along the same axis.
# - Motion and charging also stay the same: `steps = min(head_to_dest_distance,
#   grey_count)`, and the first `steps` grey cells from the head-facing side turn
#   orange.
# - The only changed branch is "what happens when the moved head actually reaches
#   the destination line". In the original task, the moved head remains maroon and
#   simply overlaps the stationary destination. In this variant, that overlap is
#   turned into black border color, so the destination line is absorbed into the
#   container as a docking seam. Because the normal container border is still one
#   cell behind it, the front side becomes a 2-cell-thick black interface.
# - Example, head on top:
#   original output after full reach: `9` line at row r, black top border at row r+1
#   variant output after full reach:  `0` line at row r, black top border at row r+1
#   The difference is visible only on complete docking; partial moves still look
#   like the original task.
#
# b) Complexity calibration
# - The variant keeps the original object vocabulary and matching logic, so there is
#   no new ambiguity in pairing heads with destinations.
# - To make D8 observable, the generator forces at least one full-docking mover in
#   every sample. Many samples also contain a partial mover or a fully orange decoy,
#   so the dataset still shows the old cap logic and the new docking branch together.
# - The docking effect has only one mode: full reach -> merged black seam. This is a
#   small, repeated rule that humans can infer from 2-4 examples.
#
# c) Old solver assumptions that matter here
# - Container detection depends only on rectangular black borders around grey/orange
#   interiors. The input format is unchanged, so this still holds.
# - Head detection assumes non-bg, non-border, non-grey, non-orange colors adjacent
#   to a border. The input still satisfies that.
# - Destination matching assumes same color and same shape as the head. The input
#   still satisfies that.
# - The original writeback assumed "destination always stays maroon" by copying the
#   input grid to `out`, erasing only the original head pixels, and then stamping the
#   moved head in maroon. That assumption is no longer valid on the full-docking
#   branch, because the overlapped destination must become black instead of staying 9.
#   The fix is local: keep the same erase/paste logic, but replace the moved head
#   stamp with a black seam when `steps == head_to_dest_distance`.
#
# d) Placement legality
# - The generator still reserves one whole assembly per container:
#   destination + gap + head + container.
# - Since the container never moves farther than the destination line, every output
#   remains inside the same reserved assembly. The docking seam occupies the same row
#   or column as the original destination, so it does not need extra space.
# - Assemblies are placed with a 1-cell background moat, keeping both input and
#   output non-overlapping and easy for the solver to parse.
#
# e) Visual observability
# - On a full dock, the front maroon line disappears into a thick black front edge.
#   This is much easier to spot than a hidden internal state change.
# - On a partial move, the maroon head stays visible, giving the human contrast:
#   "sometimes it is still a maroon head, sometimes it turns into black when it
#   reaches the goal line".
#
# f) Visual cleanliness
# - No extra markers are introduced.
# - The merged seam reuses the destination line location, so the output remains a
#   single compact object instead of stacking many symbols in one corridor.
#
# g) Color-change inferability
# - The original grey->orange charging remains and is still spatially structured.
# - The new color change is limited to a single visible condition:
#   only the fully reached head/destination line turns from maroon to black.
#   This is tied to the obvious geometric event of contact, not to a hidden value.
#
# New implicit assumptions and how the generator enforces them
# ------------------------------------------------------------
# - `main()` now assumes that at least some samples may have `steps == distance`,
#   otherwise the new branch would never be exercised. The generator guarantees at
#   least one such mover per grid.
# - The docking seam assumes the destination lies exactly on the movement axis and
#   has the same span as the head. The generator keeps the original construction, so
#   this remains true for every object.

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

# head on top -> container moves up; head on bottom -> moves down, etc.
SIDE_TO_DIR = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}


def _rect_border_coords(r0: int, c0: int, r1: int, c1: int):
    for c in range(c0, c1 + 1):
        yield (r0, c)
        yield (r1, c)
    for r in range(r0 + 1, r1):
        yield (r, c0)
        yield (r, c1)


def _interior_fill_order(inner_h: int, inner_w: int, head_side: int) -> list[tuple[int, int]]:
    """Interior cells in the same charging order used by the solver."""
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move bordered containers toward same-colored destination markers.

    Original behavior is preserved until the moment the head fully reaches the
    destination line. On that full-docking branch, the overlapped head/destination
    line is absorbed into the container as black border instead of remaining maroon.
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
        if any(int(g[r, c]) != int(BORDER) for r, c in _rect_border_coords(r0, c0, r1, c1)):
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

        tr0 = int(best_dest["tl"][0] - head_tl[0])
        tc0 = int(best_dest["tl"][1] - head_tl[1])
        dock_distance = abs(tr0) + abs(tc0)

        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        grey_coords = [
            (int(r0 + 1 + rr), int(c0 + 1 + cc))
            for rr in range(inner.shape[0])
            for cc in range(inner.shape[1])
            if int(inner[rr, cc]) == int(GREY)
        ]
        steps = min(int(dock_distance), int(len(grey_coords)))
        full_dock = steps == int(dock_distance) and steps > 0

        dr, dc = direction
        tr = int(dr * steps)
        tc = int(dc * steps)

        if direction == (0, 1):
            order = sorted(grey_coords, key=lambda p: (-p[1], p[0]))
        elif direction == (0, -1):
            order = sorted(grey_coords, key=lambda p: (p[1], -p[0]))
        elif direction == (1, 0):
            order = sorted(grey_coords, key=lambda p: (-p[0], -p[1]))
        else:  # (-1, 0)
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
            if 0 <= pr < patch.shape[0] and 0 <= pc < patch.shape[1] and int(patch[pr, pc]) == int(GREY):
                patch[pr, pc] = int(ORANGE)
        paste(out, patch, int(r0 + tr), int(c0 + tc), clip=True)

        if full_dock:
            for r, c in head_pts:
                rrr = int(r + tr)
                ccc = int(c + tc)
                if 0 <= rrr < H and 0 <= ccc < W:
                    out[rrr, ccc] = int(BORDER)
        else:
            for r, c in head_pts:
                rrr = int(r + tr)
                ccc = int(c + tc)
                if 0 <= rrr < H and 0 <= ccc < W:
                    out[rrr, ccc] = int(hc)

    return out


def _draw_head_line(grid: np.ndarray, cr: int, cc: int, oh: int, ow: int, head_side: int, color: int):
    if head_side == 0:
        grid[cr - 1, cc : cc + ow] = int(color)
    elif head_side == 1:
        grid[cr + oh, cc : cc + ow] = int(color)
    elif head_side == 2:
        grid[cr : cr + oh, cc - 1] = int(color)
    else:
        grid[cr : cr + oh, cc + ow] = int(color)


def _draw_dest_line(grid: np.ndarray, cr: int, cc: int, oh: int, ow: int, head_side: int, gap: int, color: int):
    if head_side == 0:
        grid[cr - 2 - gap, cc : cc + ow] = int(color)
    elif head_side == 1:
        grid[cr + oh + 1 + gap, cc : cc + ow] = int(color)
    elif head_side == 2:
        grid[cr : cr + oh, cc - 2 - gap] = int(color)
    else:
        grid[cr : cr + oh, cc + ow + 1 + gap] = int(color)


def _expected_output_for_spec(expected: np.ndarray, spec: dict, placement: tuple[int, int]):
    oh, ow = spec["outer_h"], spec["outer_w"]
    hs = spec["head_side"]
    steps = spec["steps"]
    distance = spec["distance"]
    inner_h, inner_w = spec["inner_h"], spec["inner_w"]
    initial_orange = spec["initial_orange"]
    dr, dc = SIDE_TO_DIR[hs]

    cr = placement[0] + spec["cont_r"]
    cc = placement[1] + spec["cont_c"]

    if steps == 0:
        return

    expected[cr : cr + oh, cc : cc + ow] = int(BG)
    _draw_head_line(expected, cr, cc, oh, ow, hs, BG)

    nr = cr + dr * steps
    nc = cc + dc * steps

    fill_order = _interior_fill_order(inner_h, inner_w, hs)
    interior = np.full((inner_h, inner_w), GREY, dtype=int)
    for idx_cell in range(initial_orange + steps):
        r, c = fill_order[idx_cell]
        interior[r, c] = int(ORANGE)
    patch = np.full((oh, ow), BORDER, dtype=int)
    patch[1:-1, 1:-1] = interior
    expected[nr : nr + oh, nc : nc + ow] = patch

    if steps == distance:
        _draw_head_line(expected, nr, nc, oh, ow, hs, BORDER)
    else:
        _draw_head_line(expected, nr, nc, oh, ow, hs, HEAD_COLOR)


def _choose_mover_modes(n_active: int) -> list[str]:
    modes = ["dock"]
    remaining = n_active - 1
    for idx in range(remaining):
        if idx == remaining - 1 and "partial" not in modes and n_active > 1:
            modes.append("partial")
        else:
            modes.append(random.choice(["dock", "partial"]))
    random.shuffle(modes)
    return modes


def _gen_attempt() -> np.ndarray | None:
    n_containers = random.randint(2, 4)
    n_decoys = random.choice([0, 0, 1])
    n_active = n_containers - n_decoys
    if n_active < 1:
        n_active = 1
        n_decoys = n_containers - 1

    active_modes = _choose_mover_modes(n_active)
    specs = []

    for idx in range(n_containers):
        is_decoy = idx >= n_active
        inner_h = random.randint(1, 8)
        inner_w = random.randint(1, 5)
        inner_area = inner_h * inner_w
        head_side = random.randint(0, 3)
        outer_h = inner_h + 2
        outer_w = inner_w + 2

        if is_decoy:
            gap = random.randint(0, 6)
            grey = 0
            initial_orange = inner_area
            steps = 0
            distance = gap + 1
            mode = "decoy"
        else:
            mode = active_modes[idx]
            if mode == "dock":
                max_gap = min(10, inner_area - 1)
                gap = random.randint(0, max_gap)
                distance = gap + 1
                grey = random.randint(distance, inner_area)
                steps = distance
            else:
                gap = random.randint(1, 10)
                distance = gap + 1
                grey = random.randint(1, min(inner_area, distance - 1))
                steps = grey
            initial_orange = inner_area - grey

        specs.append(
            {
                "inner_h": inner_h,
                "inner_w": inner_w,
                "outer_h": outer_h,
                "outer_w": outer_w,
                "head_side": head_side,
                "gap": gap,
                "distance": distance,
                "steps": steps,
                "initial_orange": initial_orange,
                "grey": grey,
                "mode": mode,
                "is_decoy": is_decoy,
            }
        )

    assemblies = []
    for spec in specs:
        oh, ow = spec["outer_h"], spec["outer_w"]
        hs = spec["head_side"]
        gap = spec["gap"]

        if hs == 0:
            total_h = 1 + gap + 1 + oh
            total_w = ow
            cont_r = gap + 2
            cont_c = 0
        elif hs == 1:
            total_h = oh + 1 + gap + 1
            total_w = ow
            cont_r = 0
            cont_c = 0
        elif hs == 2:
            total_h = oh
            total_w = 1 + gap + 1 + ow
            cont_r = 0
            cont_c = gap + 2
        else:
            total_h = oh
            total_w = ow + 1 + gap + 1
            cont_r = 0
            cont_c = 0

        assemblies.append({**spec, "total_h": total_h, "total_w": total_w, "cont_r": cont_r, "cont_c": cont_c})

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
    for assembly in assemblies:
        th, tw = assembly["total_h"], assembly["total_w"]
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
    for assembly, (pr, pc) in zip(assemblies, placements):
        oh, ow = assembly["outer_h"], assembly["outer_w"]
        hs = assembly["head_side"]
        gap = assembly["gap"]
        inner_h, inner_w = assembly["inner_h"], assembly["inner_w"]
        initial_orange = assembly["initial_orange"]

        cr = pr + assembly["cont_r"]
        cc = pc + assembly["cont_c"]

        grid[cr, cc : cc + ow] = int(BORDER)
        grid[cr + oh - 1, cc : cc + ow] = int(BORDER)
        grid[cr : cr + oh, cc] = int(BORDER)
        grid[cr : cr + oh, cc + ow - 1] = int(BORDER)

        fill_order = _interior_fill_order(inner_h, inner_w, hs)
        interior = np.full((inner_h, inner_w), GREY, dtype=int)
        for idx_cell in range(initial_orange):
            r, c = fill_order[idx_cell]
            interior[r, c] = int(ORANGE)
        grid[cr + 1 : cr + 1 + inner_h, cc + 1 : cc + 1 + inner_w] = interior

        _draw_head_line(grid, cr, cc, oh, ow, hs, HEAD_COLOR)
        _draw_dest_line(grid, cr, cc, oh, ow, hs, gap, HEAD_COLOR)

    try:
        output = main(grid)
    except Exception:
        return None

    expected = grid.copy()
    for assembly, placement in zip(assemblies, placements):
        _expected_output_for_spec(expected, assembly, placement)

    if not np.array_equal(output, expected):
        return None

    if not any(a["steps"] == a["distance"] and a["steps"] > 0 for a in assemblies):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid after 300 attempts")
