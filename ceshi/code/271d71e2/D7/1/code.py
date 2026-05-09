# Variant of 271d71e2
# Dimension: D7 — 干扰项（decoy）设计 (obstacle)
# Change: 0-1个全橙色容器作为干扰项，不移动（灰色像素数=0，步数=0）... → 无目标线的容器作为干扰项

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, bbox_coords, colors_present, components_mask, paste

# Design analysis for the D7 variant
# source_task: None
#
# a) Concrete behavior and trace:
# - Objects stay visually the same as the base task: magenta background, black rectangular
#   containers, grey/orange interior, and a maroon head-line flush with one container side.
# - Movers still have a same-shaped maroon destination line farther along the head direction.
# - The new decoy also has a head and grey cells, but no destination line at all. It stays still
#   in the output even though it still looks "chargeable".
# - Small trace: if a 3x5 container has a top head and three grey cells, base logic would scan
#   for a matching maroon line above it and move upward by min(distance, 3). In the variant,
#   if no such line exists, the container must remain unchanged rather than being inferred from
#   its grey/orange count. The changed step is the destination-detection stage, not the charge
#   or translation stage.
#
# b) Complexity calibration:
# - Decoys still share the same colors and head shape as movers, so they cannot be identified by
#   a trivial "all orange means decoy" shortcut.
# - At most one decoy is kept, matching the original task scale, but the generator biases toward
#   including one so the variant is visible in a few examples.
# - Decoys reserve empty space in front of the head to make "missing destination" visually clear
#   instead of looking truncated by the grid edge.
#
# c) Hidden assumptions in the original solver that must change:
# - Original external destination matching accepted any same-color, same-shape component. That is
#   safe only when every such component is a destination for some container. In this variant,
#   another container's head would become a false destination candidate.
# - The fix is to treat only non-border-adjacent maroon components as external destinations.
#   Heads always touch a black border; standalone destinations do not. Touching head+dest cases
#   (gap=0) are still handled by splitting the shared component and keeping only the non-border
#   side as destination.
# - Original "if no dests: continue" remains valid and now becomes the decoy behavior.
#
# d) Placement legality:
# - Assemblies are still separated by at least one background cell.
# - Movers reserve the original destination corridor; decoys reserve a forward empty corridor with
#   no destination line. This keeps all objects in-bounds and prevents output overlaps because all
#   motion stays inside the mover assembly footprint.
# - The generator re-runs the same analyzer used by main() and rejects any layout where a decoy is
#   accidentally paired with a destination or a mover is paired with the wrong one.
#
# e) Visual observability:
# - Human-visible difference: some containers with grey cells and a head do move, others with the
#   same local structure do not, and the deciding cue is the presence/absence of the distant line.
# - Keeping decoy heads away from the grid edge avoids the ambiguous interpretation that the target
#   line may simply be cropped out.
#
# f) Visual cleanliness:
# - No extra colors or overlays are added.
# - Containers remain axis-aligned rectangles with simple straight-line heads/targets.
# - A one-cell global buffer between assemblies keeps the scenes readable.
#
# g) Color inferability:
# - No new color mapping is introduced. Maroon still means head/target markers, black is border,
#   grey/orange are charge state.
#
# Post-change solver assumptions that the generator must satisfy:
# - Every true mover must have exactly one nearest aligned non-border-adjacent target line.
# - Every decoy must have none.
# - Heads must remain the only maroon pixels adjacent to their own border.

BG = 6
BORDER = 0
GREY = 5
ORANGE = 7
HEAD_COLOR = 9
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# head on top -> container moves up; head on bottom -> moves down, etc.
SIDE_TO_DIR = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}


def _dominant_non_border_bg(g: np.ndarray) -> int:
    cnt = Counter(map(int, g.flatten()))
    return int(max((c for c in cnt if int(c) != BORDER), key=lambda c: int(cnt[int(c)])))


def _rect_border_coords(r0: int, c0: int, r1: int, c1: int):
    for c in range(c0, c1 + 1):
        yield (r0, c)
        yield (r1, c)
    for r in range(r0 + 1, r1):
        yield (r, c0)
        yield (r, c1)


def _touches_border(g: np.ndarray, pts: set[tuple[int, int]]) -> bool:
    h, w = g.shape
    for r, c in pts:
        for dr, dc in DIR4:
            rr, cc = int(r + dr), int(c + dc)
            if 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == BORDER:
                return True
    return False


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


def _ordered_greys(grey_coords: list[tuple[int, int]], direction: tuple[int, int]) -> list[tuple[int, int]]:
    if direction == (0, 1):
        return sorted(grey_coords, key=lambda p: (-p[1], p[0]))
    if direction == (0, -1):
        return sorted(grey_coords, key=lambda p: (p[1], -p[0]))
    if direction == (1, 0):
        return sorted(grey_coords, key=lambda p: (-p[0], -p[1]))
    return sorted(grey_coords, key=lambda p: (p[0], p[1]))


def _find_containers(g: np.ndarray) -> list[dict]:
    h, w = g.shape
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
    containers.sort(key=lambda d: d["box"])
    return containers


def _collect_marker_components(g: np.ndarray, bg: int) -> dict[int, list[dict]]:
    comp_by_color: dict[int, list[dict]] = defaultdict(list)
    marker_colors = [c for c in colors_present(g, exclude={bg, BORDER, GREY, ORANGE})]
    for col in marker_colors:
        labc, nc = label(g == int(col), structure=STRUCT4)
        for k in range(1, int(nc) + 1):
            coords = np.argwhere(labc == int(k))
            if coords.size == 0:
                continue
            r0, c0, _r1, _c1 = bbox_coords(coords)
            pts = {(int(r), int(c)) for r, c in coords}
            shape = {(int(r - r0), int(c - c0)) for r, c in pts}
            comp_by_color[int(col)].append(
                {
                    "pts": pts,
                    "shape": shape,
                    "tl": (int(r0), int(c0)),
                    "touches_border": _touches_border(g, pts),
                }
            )
    return comp_by_color


def _pick_head_for_container(
    box_: tuple[int, int, int, int],
    g: np.ndarray,
    bg: int,
    comp_by_color: dict[int, list[dict]],
) -> dict | None:
    h, w = g.shape
    r0, c0, r1, c1 = box_
    head_seeds = []
    for r, c in _rect_border_coords(r0, c0, r1, c1):
        for dr, dc in DIR4:
            rr, cc = int(r + dr), int(c + dc)
            if 0 <= rr < h and 0 <= cc < w:
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
        head_side = 2
    elif hc0 > c1:
        direction = (0, 1)
        head_side = 3
    elif hr1 < r0:
        direction = (-1, 0)
        head_side = 0
    elif hr0 > r1:
        direction = (1, 0)
        head_side = 1
    else:
        direction = (0, 0)
        head_side = -1

    r00, c00 = int(hr0), int(hc0)
    head_shape = {(int(r - r00), int(c - c00)) for r, c in head_pts}
    return {
        "color": hc,
        "head_pts": head_pts,
        "head_shape": head_shape,
        "head_tl": (r00, c00),
        "comp_pts": set(comp0["pts"]),
        "dir": direction,
        "head_side": head_side,
    }


def _aligned_distance(direction: tuple[int, int], head_tl: tuple[int, int], dest_tl: tuple[int, int]) -> int | None:
    tr0 = int(dest_tl[0] - head_tl[0])
    tc0 = int(dest_tl[1] - head_tl[1])
    if direction[0] != 0 and tc0 != 0:
        return None
    if direction[1] != 0 and tr0 != 0:
        return None
    if direction[0] == 1 and tr0 <= 0:
        return None
    if direction[0] == -1 and tr0 >= 0:
        return None
    if direction[1] == 1 and tc0 <= 0:
        return None
    if direction[1] == -1 and tc0 >= 0:
        return None
    return abs(tr0) + abs(tc0)


def _corridor_clear(
    g: np.ndarray,
    bg: int,
    head_pts: set[tuple[int, int]],
    direction: tuple[int, int],
    distance: int,
) -> bool:
    for r, c in head_pts:
        for step in range(1, int(distance)):
            rr = int(r + direction[0] * step)
            cc = int(c + direction[1] * step)
            if int(g[rr, cc]) != bg:
                return False
    return True


def _find_best_destination(
    box_: tuple[int, int, int, int],
    g: np.ndarray,
    bg: int,
    comp_by_color: dict[int, list[dict]],
    head_info: dict,
) -> dict | None:
    del box_
    hc = int(head_info["color"])
    head_shape = head_info["head_shape"]
    head_tl = head_info["head_tl"]
    comp_pts = set(head_info["comp_pts"])
    head_pts = set(head_info["head_pts"])
    direction = head_info["dir"]
    if direction == (0, 0):
        return None

    dests = []
    seen = set()

    for d in comp_by_color.get(hc, []):
        if d["shape"] != head_shape or d["pts"] == comp_pts or d["touches_border"]:
            continue
        dist = _aligned_distance(direction, head_tl, d["tl"])
        if dist is None:
            continue
        if not _corridor_clear(g, bg, head_pts, direction, dist):
            continue
        key = tuple(sorted(d["pts"]))
        if key not in seen:
            seen.add(key)
            dests.append(d)

    rem = comp_pts - head_pts
    if rem:
        h, w = g.shape
        mask_rem = np.zeros((h, w), dtype=bool)
        for rr, cc in rem:
            mask_rem[int(rr), int(cc)] = True
        labr, nr = label(mask_rem, structure=STRUCT4)
        for k in range(1, int(nr) + 1):
            pts = {(int(r), int(c)) for r, c in np.argwhere(labr == int(k))}
            if not pts or _touches_border(g, pts):
                continue
            rs = [r for r, _c in pts]
            cs = [c for _r, c in pts]
            r00, c00 = int(min(rs)), int(min(cs))
            shp = {(int(r - r00), int(c - c00)) for r, c in pts}
            if shp != head_shape:
                continue
            dist = _aligned_distance(direction, head_tl, (r00, c00))
            if dist is None:
                continue
            key = tuple(sorted(pts))
            if key not in seen:
                seen.add(key)
                dests.append({"pts": pts, "shape": shp, "tl": (r00, c00), "touches_border": False})

    best_dist = None
    best_dest = None
    for d in dests:
        dist = _aligned_distance(direction, head_tl, d["tl"])
        if dist is None:
            continue
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_dest = d
    return best_dest


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    bg = _dominant_non_border_bg(g)
    containers = _find_containers(g)
    assert containers
    comp_by_color = _collect_marker_components(g, bg)

    infos = []
    for cont in containers:
        box_ = cont["box"]
        r0, c0, r1, c1 = box_
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        grey_coords = [
            (int(r0 + 1 + rr), int(c0 + 1 + cc))
            for rr in range(inner.shape[0])
            for cc in range(inner.shape[1])
            if int(inner[rr, cc]) == GREY
        ]
        head_info = _pick_head_for_container(box_, g, bg, comp_by_color)
        best_dest = None
        head_dist = None
        steps = 0
        if head_info is not None and head_info["dir"] != (0, 0):
            best_dest = _find_best_destination(box_, g, bg, comp_by_color, head_info)
            if best_dest is not None:
                head_dist = _aligned_distance(head_info["dir"], head_info["head_tl"], best_dest["tl"])
                steps = min(int(head_dist), int(len(grey_coords)))
        infos.append(
            {
                "box": box_,
                "grey_coords": grey_coords,
                "head_info": head_info,
                "dest": best_dest,
                "has_dest": best_dest is not None,
                "distance": head_dist,
                "steps": steps,
            }
        )
    return {"grid": g, "bg": bg, "containers": infos}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move only the containers whose head has a matching destination line.

    Decoys look like ordinary chargeable containers except that the destination line is absent.
    They keep their head and grey pixels unchanged.
    """
    analysis = _analyze_grid(grid)
    g = analysis["grid"]
    bg = int(analysis["bg"])
    out = g.copy()

    for cont in analysis["containers"]:
        head_info = cont["head_info"]
        if head_info is None or cont["dest"] is None or int(cont["steps"]) <= 0:
            continue

        r0, c0, r1, c1 = cont["box"]
        hc = int(head_info["color"])
        head_pts = set(head_info["head_pts"])
        direction = tuple(head_info["dir"])
        steps = int(cont["steps"])
        dr, dc = direction
        tr = int(dr * steps)
        tc = int(dc * steps)

        to_fill = set(_ordered_greys(cont["grey_coords"], direction)[:steps])

        out[r0 : r1 + 1, c0 : c1 + 1] = bg
        for r, c in head_pts:
            if int(out[int(r), int(c)]) == hc:
                out[int(r), int(c)] = bg

        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        for rr, cc in to_fill:
            pr = int(rr - r0)
            pc = int(cc - c0)
            if int(patch[pr, pc]) == GREY:
                patch[pr, pc] = ORANGE
        paste(out, patch, int(r0 + tr), int(c0 + tc), clip=True)

        for r, c in head_pts:
            rr = int(r + tr)
            cc = int(c + tc)
            if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
                out[rr, cc] = hc

    return out


def _build_specs() -> list[dict]:
    n_containers = random.randint(2, 4)
    n_decoys = 1 if (n_containers >= 2 and random.random() < 0.7) else 0
    specs = []
    for i in range(n_containers):
        is_decoy = i < n_decoys
        inner_h = random.randint(1, 8)
        inner_w = random.randint(1, 5)
        inner_area = inner_h * inner_w
        head_side = random.randint(0, 3)
        outer_h = inner_h + 2
        outer_w = inner_w + 2
        grey = random.randint(1, inner_area)
        initial_orange = inner_area - grey
        if is_decoy:
            forward_pad = random.randint(2, 5)
            gap = None
            steps = 0
        else:
            gap = random.randint(0, 8)
            forward_pad = None
            steps = min(grey, gap + 1)
        specs.append(
            {
                "inner_h": inner_h,
                "inner_w": inner_w,
                "outer_h": outer_h,
                "outer_w": outer_w,
                "head_side": head_side,
                "grey": grey,
                "initial_orange": initial_orange,
                "gap": gap,
                "steps": steps,
                "forward_pad": forward_pad,
                "is_decoy": is_decoy,
            }
        )
    random.shuffle(specs)
    return specs


def _build_assembly(spec: dict) -> dict:
    oh, ow = int(spec["outer_h"]), int(spec["outer_w"])
    hs = int(spec["head_side"])
    if spec["is_decoy"]:
        pad = int(spec["forward_pad"])
        if hs == 0:
            total_h, total_w = pad + 1 + oh, ow
            cont_r, cont_c = pad + 1, 0
        elif hs == 1:
            total_h, total_w = oh + 1 + pad, ow
            cont_r, cont_c = 0, 0
        elif hs == 2:
            total_h, total_w = oh, pad + 1 + ow
            cont_r, cont_c = 0, pad + 1
        else:
            total_h, total_w = oh, ow + 1 + pad
            cont_r, cont_c = 0, 0
    else:
        gap = int(spec["gap"])
        if hs == 0:
            total_h, total_w = 1 + gap + 1 + oh, ow
            cont_r, cont_c = gap + 2, 0
        elif hs == 1:
            total_h, total_w = oh + 1 + gap + 1, ow
            cont_r, cont_c = 0, 0
        elif hs == 2:
            total_h, total_w = oh, 1 + gap + 1 + ow
            cont_r, cont_c = 0, gap + 2
        else:
            total_h, total_w = oh, ow + 1 + gap + 1
            cont_r, cont_c = 0, 0
    return {**spec, "total_h": total_h, "total_w": total_w, "cont_r": cont_r, "cont_c": cont_c}


def _draw_input(grid: np.ndarray, assemblies: list[dict], placements: list[tuple[int, int]]) -> dict:
    expected_by_box = {}
    for a, (pr, pc) in zip(assemblies, placements):
        oh, ow = int(a["outer_h"]), int(a["outer_w"])
        hs = int(a["head_side"])
        inner_h, inner_w = int(a["inner_h"]), int(a["inner_w"])
        initial_orange = int(a["initial_orange"])
        cr = int(pr + a["cont_r"])
        cc = int(pc + a["cont_c"])

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
        elif hs == 1:
            grid[cr + oh, cc : cc + ow] = HEAD_COLOR
        elif hs == 2:
            grid[cr : cr + oh, cc - 1] = HEAD_COLOR
        else:
            grid[cr : cr + oh, cc + ow] = HEAD_COLOR

        if not a["is_decoy"]:
            gap = int(a["gap"])
            if hs == 0:
                grid[cr - 2 - gap, cc : cc + ow] = HEAD_COLOR
            elif hs == 1:
                grid[cr + oh + 1 + gap, cc : cc + ow] = HEAD_COLOR
            elif hs == 2:
                grid[cr : cr + oh, cc - 2 - gap] = HEAD_COLOR
            else:
                grid[cr : cr + oh, cc + ow + 1 + gap] = HEAD_COLOR

        expected_by_box[(cr, cc, cr + oh - 1, cc + ow - 1)] = a
    return expected_by_box


def _place_assemblies(assemblies: list[dict]) -> tuple[int, int, list[tuple[int, int]]] | None:
    total_area = sum(int(a["total_h"]) * int(a["total_w"]) for a in assemblies)
    min_side = max(8, int(total_area ** 0.5) + 2)
    max_side = min(26, min_side + 8)
    max_asm_h = max(int(a["total_h"]) for a in assemblies)
    max_asm_w = max(int(a["total_w"]) for a in assemblies)

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
        th, tw = int(a["total_h"]), int(a["total_w"])
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
    return grid_h, grid_w, placements


def _expected_output(grid: np.ndarray, assemblies: list[dict], placements: list[tuple[int, int]]) -> np.ndarray:
    expected = grid.copy()
    for a, (pr, pc) in zip(assemblies, placements):
        if a["is_decoy"] or int(a["steps"]) <= 0:
            continue

        oh, ow = int(a["outer_h"]), int(a["outer_w"])
        hs = int(a["head_side"])
        dr, dc = SIDE_TO_DIR[hs]
        inner_h, inner_w = int(a["inner_h"]), int(a["inner_w"])
        initial_orange = int(a["initial_orange"])
        steps = int(a["steps"])
        cr = int(pr + a["cont_r"])
        cc = int(pc + a["cont_c"])

        expected[cr : cr + oh, cc : cc + ow] = BG
        if hs == 0:
            expected[cr - 1, cc : cc + ow] = BG
        elif hs == 1:
            expected[cr + oh, cc : cc + ow] = BG
        elif hs == 2:
            expected[cr : cr + oh, cc - 1] = BG
        else:
            expected[cr : cr + oh, cc + ow] = BG

        nr = int(cr + dr * steps)
        nc = int(cc + dc * steps)
        fill_order = _interior_fill_order(inner_h, inner_w, hs)
        interior = np.full((inner_h, inner_w), GREY, dtype=int)
        for idx_cell in range(initial_orange + steps):
            r, c = fill_order[idx_cell]
            interior[r, c] = ORANGE
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
    assemblies = [_build_assembly(spec) for spec in specs]
    placement_info = _place_assemblies(assemblies)
    if placement_info is None:
        return None
    grid_h, grid_w, placements = placement_info
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    expected_by_box = _draw_input(grid, assemblies, placements)

    try:
        analysis = _analyze_grid(grid)
    except Exception:
        return None

    if len(analysis["containers"]) != len(assemblies):
        return None

    seen_decoy = False
    for info in analysis["containers"]:
        spec = expected_by_box.get(info["box"])
        if spec is None:
            return None
        head_info = info["head_info"]
        if head_info is None or int(head_info["head_side"]) != int(spec["head_side"]):
            return None
        if spec["is_decoy"]:
            seen_decoy = True
            if info["has_dest"] or int(len(info["grey_coords"])) <= 0:
                return None
        else:
            if not info["has_dest"]:
                return None
            if int(info["distance"]) != int(spec["gap"]) + 1:
                return None
            if int(info["steps"]) != int(spec["steps"]):
                return None

    expected = _expected_output(grid, assemblies, placements)
    try:
        output = main(grid)
    except Exception:
        return None
    if not np.array_equal(output, expected):
        return None

    if any(spec["is_decoy"] for spec in specs) and not seen_decoy:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid after 500 attempts")
