# Combo variant of 271d71e2 (index 12)
# combo_spec: D8/1+D14/1
# Name: 对接合并留影子
#
# Design analysis
# ===============
# a) Concrete behavior and trace
# - Input grammar stays the same as the base task: magenta background, black
#   rectangular container, grey/orange interior, one maroon head line adjacent to
#   one edge, and one matching maroon destination line farther along the movement
#   axis.
# - Motion and charging also stay the same:
#   `steps = min(head_to_dest_distance, grey_count)`.
# - The fusion changes only the output writeback:
#   1. If `steps > 0`, the source container position does not disappear fully;
#      its old outer rectangle is redrawn as a hollow grey outline.
#   2. The real container is pasted at the translated position with `steps` more
#      grey cells charged to orange.
#   3. If the translated head fully reaches the destination line
#      (`steps == distance`), the head/destination overlap becomes black instead of
#      maroon, creating a docking seam. Otherwise the moved head stays maroon.
# - Small trace, moving upward by 2:
#   original: clear old box/head -> paste moved box -> paste moved maroon head.
#   fused:    clear old box/head -> draw old grey outline shadow -> paste moved
#             box -> paste black seam on the front edge if fully docked, else paste
#             moved maroon head.
# - Visible difference from the base task:
#   the output can simultaneously show a grey source-position shell and a docked
#   final object whose front marker has turned into black border.
#
# b) Complexity calibration
# - The fusion still uses the original object vocabulary and original matching
#   logic. There is only one new trail artifact and one new docking branch.
# - To keep the rule learnable, the generator guarantees at least one full dock
#   with distance >= 2 so both "black docking seam" and "visible source shadow"
#   appear clearly in the same sample.
# - When the grid has at least two active movers, the generator also includes a
#   partial mover, so examples contrast the two front-edge outcomes:
#   full reach -> black seam, partial reach -> maroon head.
#
# c) Implicit-assumption audit
# - Container detection still assumes the input has only black-bordered containers.
#   The grey shadow is output-only, so this assumption remains valid.
# - Head detection still assumes non-bg/non-border/non-grey/non-orange pixels touch
#   a container border. Input construction preserves that.
# - Destination matching still assumes the destination has the same color and shape
#   as the head. Input construction preserves that.
# - The base writeback assumed the source vanished and the destination stayed
#   maroon. Both are broken by the fusion:
#   source must be redrawn as a grey outline after clearing, and a full dock must
#   overwrite the destination row/column in black.
# - The paste order now matters: draw shadow before stamping the moved container so
#   the real container occludes any overlapping part of the shadow.
#
# d) Placement legality
# - Each object still lives inside the original reserved assembly:
#   destination + gap + head + container.
# - The moved container never goes past the destination line, so the final object
#   stays inside the same reserved footprint.
# - The shadow occupies exactly the source box footprint, which is also already
#   inside the reserved assembly, so it does not need extra global space.
# - Assemblies are placed with a one-cell background moat, which keeps different
#   objects separated in both input and output.
#
# e) Visual observability
# - Humans can see two repeated cues:
#   moved objects leave a grey hollow shell behind, and only fully reaching objects
#   lose the maroon head in favor of a black docking edge.
# - These cues depend only on visible geometry: whether the object reaches the goal
#   line and where it started from. No hidden ordering or numeric color-value logic
#   is required.
#
# f) Visual cleanliness
# - The shadow is hollow and headless, so it is distinguishable from both the real
#   container and normal grey interior cells.
# - The docking effect reuses the destination line instead of adding extra marks,
#   so the final object remains compact rather than cluttered.
#
# g) Change inferability
# - The only color changes are structured:
#   grey interior -> orange follows the original spatial charging order;
#   maroon front line -> black happens only on visible contact with the destination;
#   source border -> grey outline happens only where a moved container used to be.
# - The generator guarantees non-degenerate samples where these signals matter, so
#   the combo does not collapse back to the base task visually.
#
# New solver assumptions after the fusion
# ---------------------------------------
# - At least one sample should exercise the full-docking branch, otherwise D8 is
#   invisible. The generator guarantees one such mover.
# - At least one moved object should have `steps >= 2`, otherwise the source shadow
#   can be too subtle. The generator makes the guaranteed full dock use distance >= 2.
# - The solver still relies on one head/destination pair per active container and
#   on clear movement corridors. The generator constructs isolated assemblies to
#   enforce that.

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import random
import sys

import numpy as np
from scipy.ndimage import label

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from arc_utils import STRUCT4, bbox_coords, colors_present, components_mask, paste


BG = 6
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


def _draw_rect_outline(arr: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int):
    arr[r0, c0 : c1 + 1] = int(color)
    arr[r1, c0 : c1 + 1] = int(color)
    arr[r0 : r1 + 1, c0] = int(color)
    arr[r0 : r1 + 1, c1] = int(color)


def _draw_head_line(arr: np.ndarray, cr: int, cc: int, oh: int, ow: int, head_side: int, color: int):
    if head_side == 0:
        arr[cr - 1, cc : cc + ow] = int(color)
    elif head_side == 1:
        arr[cr + oh, cc : cc + ow] = int(color)
    elif head_side == 2:
        arr[cr : cr + oh, cc - 1] = int(color)
    else:
        arr[cr : cr + oh, cc + ow] = int(color)


def _draw_dest_line(arr: np.ndarray, cr: int, cc: int, oh: int, ow: int, head_side: int, gap: int, color: int):
    if head_side == 0:
        arr[cr - 2 - gap, cc : cc + ow] = int(color)
    elif head_side == 1:
        arr[cr + oh + 1 + gap, cc : cc + ow] = int(color)
    elif head_side == 2:
        arr[cr : cr + oh, cc - 2 - gap] = int(color)
    else:
        arr[cr : cr + oh, cc + ow + 1 + gap] = int(color)


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


def _grey_fill_order(grey_coords: list[tuple[int, int]], direction: tuple[int, int]) -> list[tuple[int, int]]:
    if direction == (0, 1):
        return sorted(grey_coords, key=lambda p: (-p[1], p[0]))
    if direction == (0, -1):
        return sorted(grey_coords, key=lambda p: (p[1], -p[0]))
    if direction == (1, 0):
        return sorted(grey_coords, key=lambda p: (-p[0], -p[1]))
    return sorted(grey_coords, key=lambda p: (p[0], p[1]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move bordered containers toward same-colored destination markers.

    Fusion behavior:
    - movement and grey->orange charging stay as in the base task
    - moved containers leave a grey outline shadow at their source box
    - fully docked heads merge with the destination line as black border
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
        to_fill = set(_grey_fill_order(grey_coords, direction)[:steps])

        out[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
        for r, c in head_pts:
            if int(out[int(r), int(c)]) == int(hc):
                out[int(r), int(c)] = int(bg)

        if steps > 0:
            _draw_rect_outline(out, r0, c0, r1, c1, GREY)

        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        for rr, cc in to_fill:
            pr = int(rr - r0)
            pc = int(cc - c0)
            if 0 <= pr < patch.shape[0] and 0 <= pc < patch.shape[1] and int(patch[pr, pc]) == int(GREY):
                patch[pr, pc] = int(ORANGE)
        paste(out, patch, int(r0 + tr), int(c0 + tc), clip=True)

        head_color = BORDER if full_dock else hc
        for r, c in head_pts:
            rrr = int(r + tr)
            ccc = int(c + tc)
            if 0 <= rrr < H and 0 <= ccc < W:
                out[rrr, ccc] = int(head_color)

    return out


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
    _draw_rect_outline(expected, cr, cc, cr + oh - 1, cc + ow - 1, GREY)

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

    head_color = BORDER if steps == distance else HEAD_COLOR
    _draw_head_line(expected, nr, nc, oh, ow, hs, head_color)


def _choose_active_modes(n_active: int) -> list[str]:
    if n_active <= 0:
        return []
    modes = ["dock_far"]
    if n_active >= 2:
        modes.append("partial")
    while len(modes) < n_active:
        modes.append(random.choice(["dock_far", "partial"]))
    random.shuffle(modes)
    return modes


def _sample_active_spec(mode: str) -> dict:
    while True:
        inner_h = random.randint(1, 8)
        inner_w = random.randint(1, 5)
        inner_area = inner_h * inner_w
        head_side = random.randint(0, 3)
        outer_h = inner_h + 2
        outer_w = inner_w + 2

        if mode == "dock_far":
            if inner_area < 2:
                continue
            distance = random.randint(2, min(11, inner_area))
            grey = random.randint(distance, inner_area)
            gap = distance - 1
            steps = distance
        else:
            grey = random.randint(1, min(inner_area, 10))
            distance = random.randint(grey + 1, 11)
            gap = distance - 1
            steps = grey

        return {
            "inner_h": inner_h,
            "inner_w": inner_w,
            "outer_h": outer_h,
            "outer_w": outer_w,
            "head_side": head_side,
            "gap": gap,
            "distance": distance,
            "steps": steps,
            "grey": grey,
            "initial_orange": inner_area - grey,
            "mode": mode,
            "is_decoy": False,
        }


def _sample_decoy_spec() -> dict:
    inner_h = random.randint(1, 8)
    inner_w = random.randint(1, 5)
    inner_area = inner_h * inner_w
    head_side = random.randint(0, 3)
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    gap = random.randint(0, 6)
    return {
        "inner_h": inner_h,
        "inner_w": inner_w,
        "outer_h": outer_h,
        "outer_w": outer_w,
        "head_side": head_side,
        "gap": gap,
        "distance": gap + 1,
        "steps": 0,
        "grey": 0,
        "initial_orange": inner_area,
        "mode": "decoy",
        "is_decoy": True,
    }


def _assembly_from_spec(spec: dict) -> dict:
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

    return {**spec, "total_h": total_h, "total_w": total_w, "cont_r": cont_r, "cont_c": cont_c}


def _gen_attempt() -> np.ndarray | None:
    n_containers = random.randint(2, 4)
    min_active = min(2, n_containers)
    n_decoys = random.choice([0, 0, 1])
    n_active = n_containers - n_decoys
    if n_active < min_active:
        n_active = min_active
        n_decoys = n_containers - n_active

    active_modes = _choose_active_modes(n_active)
    specs = [_sample_active_spec(mode) for mode in active_modes]
    specs.extend(_sample_decoy_spec() for _ in range(n_decoys))
    random.shuffle(specs)

    assemblies = [_assembly_from_spec(spec) for spec in specs]

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

    if not any(a["steps"] == a["distance"] and a["steps"] >= 2 for a in assemblies):
        return None
    if not any(a["steps"] > 0 for a in assemblies):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid after 300 attempts")
