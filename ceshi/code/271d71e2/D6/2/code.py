# Variant of 271d71e2
# Dimension: D6 — 容器与目标的基数关系 (meta)
# Change: 1:1 — 每个容器恰好对应一条目标线，颜色相同（均为maroon/9）... → 颜色匹配配对

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, bbox_coords, colors_present, components_mask, paste

# Design analysis for the D6 variant
# ----------------------------------
# Source-task review:
# - No separate source_task was provided, so the only baseline is the original 271d71e2 solver/spec.
#
# a) Concretization and behavior trace
# - Geometry is unchanged: magenta background, black rectangular containers, grey/orange interiors,
#   and 1-cell-thick head/destination lines aligned with one container side.
# - The only semantic change is that head/destination markers are no longer globally maroon.
#   Each container owns a marker color chosen from a small visible palette, and its destination line
#   has the same color. Different containers in one grid usually use different marker colors.
# - Small trace:
#   Original: two containers both use 9; solver can treat every eligible 9-line as a destination pool.
#   Variant: one container may use blue(1), another green(3). The blue head must ignore the green
#   target even if it is otherwise a legal line. Movement distance and grey->orange charging are still
#   computed exactly as before after the color-matched destination is found.
#
# b) Complexity calibration
# - The variant should be visually obvious but not combinatorially noisy.
# - The palette is restricted to {1,2,3,4,8,9}; border/bg/interior colors remain unchanged.
# - Each grid has 2-4 containers and at least 2 distinct marker colors, with at least one non-9 color.
# - Within one grid, colors are unique per container. This keeps D6 focused on color pairing rather
#   than introducing a second, unnecessary same-color-multi-target ambiguity dimension.
#
# c) Implicit-assumption audit of main()
# - Original code was already partly color-generic: it scanned all non-bg/non-border/non-interior
#   colors and grouped components by color. That part survives unchanged in spirit.
# - The risky old assumption was in pick_head_for_container():
#   it selected the most common adjacent color and assumed that was the head. In the original task,
#   that was harmless because every marker was 9. In the variant, a more robust rule is needed:
#   a legal head candidate must both touch the container border and have a same-colored, same-shaped,
#   correctly aligned destination somewhere in its movement direction.
# - Erasing/stamping logic must use the per-container marker color, not a fixed constant.
#
# d) Placement legality
# - Generator still constructs standard "assemblies": destination line, gap, head line, container.
# - Assemblies are separated by at least one background cell and kept fully in-bounds.
# - Since each assembly remains self-contained and aligned exactly like the original task, the solver's
#   rectangle detection and destination search remain valid.
#
# e) Visual observability
# - Across 2-4 I/O examples, the human can see that black containers move exactly as before, but the
#   relevant target line is the one sharing the head's color. This is directly readable from pixels:
#   blue head -> blue target, green head -> green target, etc.
# - The generator rejects degenerate all-9 cases and always includes multiple marker colors.
#
# f) Visual cleanliness
# - No extra overlays or auxiliary symbols are introduced.
# - Marker colors occupy the same thin lines as before, so the scene stays clean and close to the
#   original task's readability.
#
# g) Color-change inferability
# - The color palette is small and reused across samples.
# - Colors are not used for hidden ordering or arithmetic, only for visible head<->target matching.
#
# New main() assumptions introduced by the variant and how generate_input() guarantees them
# - Assumption: every container has exactly one adjacent marker color that also yields a same-colored,
#   same-shaped aligned destination. Generator guarantees this by giving each assembly one head line
#   and one destination line of the same chosen marker color.
# - Assumption: marker colors do not collide with bg/border/grey/orange. Generator samples only from
#   MARKER_COLORS below.
# - Assumption: object groups remain spatially separated so a container does not accidentally touch a
#   foreign marker. Generator keeps a 1-cell background moat around each assembly.


BG = 6
BORDER = 0
GREY = 5
ORANGE = 7
MARKER_COLORS = [1, 2, 3, 4, 8, 9]

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Direction mapping: head_side -> (dr, dc) movement direction.
SIDE_TO_DIR = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move bordered containers toward color-matched destination markers, charging grey cells to orange.

    Variant rule:
    - Each head line may use a different visible color.
    - The matching destination is the same-shaped line of the same color lying along the movement axis.
    - Once matched, movement distance and grey->orange conversion are identical to the original task.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    cnt = Counter(map(int, g.flatten()))
    bg = int(max((c for c in cnt.keys() if int(c) != int(BORDER)), key=lambda c: int(cnt[int(c)])))

    def rect_border_coords(r0: int, c0: int, r1: int, c1: int):
        for c in range(c0, c1 + 1):
            yield (r0, c)
            yield (r1, c)
        for r in range(r0 + 1, r1):
            yield (r, c0)
            yield (r, c1)

    def split_head_from_component(box_: tuple[int, int, int, int], comp_pts: set[tuple[int, int]]):
        r0, c0, r1, c1 = box_
        border_pts = list(rect_border_coords(r0, c0, r1, c1))
        best_d = None
        dist = {}
        for pr, pc in comp_pts:
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
        if direction == (0, 0):
            return None

        head_tl = (int(hr0), int(hc0))
        head_shape = {(int(r - hr0), int(c - hc0)) for r, c in head_pts}
        return {
            "head_pts": head_pts,
            "head_shape": head_shape,
            "head_tl": head_tl,
            "dir": direction,
        }

    # Detect black-bordered containers around GREY/ORANGE interiors.
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
    marker_colors = [c for c in colors_present(g, exclude={bg, BORDER, GREY, ORANGE})]

    comp_by_color: dict[int, list[dict]] = defaultdict(list)
    for col in marker_colors:
        labc, nc = label(g == int(col), structure=STRUCT4)
        for k in range(1, int(nc) + 1):
            coords = np.argwhere(labc == int(k))
            if coords.size == 0:
                continue
            r0, c0, _r1, _c1 = bbox_coords(coords)
            pts = {(int(r), int(c)) for r, c in coords}
            shape = {(int(r - r0), int(c - c0)) for r, c in pts}
            comp_by_color[int(col)].append({"pts": pts, "shape": shape, "tl": (int(r0), int(c0))})

    def find_best_dest(
        box_: tuple[int, int, int, int],
        color: int,
        comp_pts: set[tuple[int, int]],
        head_pts: set[tuple[int, int]],
        head_shape: set[tuple[int, int]],
        head_tl: tuple[int, int],
        direction: tuple[int, int],
    ):
        dests = []
        for comp in comp_by_color.get(int(color), []):
            if comp["shape"] != head_shape:
                continue
            if comp["pts"] == comp_pts:
                continue
            dests.append(comp)

        # If gap=0, head and destination can be one connected component. Split the remainder.
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
                adj_border = False
                for r, c in pts:
                    for dr, dc in DIR4:
                        rr, cc = int(r + dr), int(c + dc)
                        if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == BORDER:
                            adj_border = True
                            break
                    if adj_border:
                        break
                if adj_border:
                    continue
                rs = [r for r, _c in pts]
                cs = [c for _r, c in pts]
                r00, c00 = int(min(rs)), int(min(cs))
                shp = {(int(r - r00), int(c - c00)) for r, c in pts}
                if shp != head_shape:
                    continue
                dests.append({"pts": pts, "shape": shp, "tl": (r00, c00)})

        best_dist = None
        best_dest = None
        for dest in dests:
            tr0 = int(dest["tl"][0] - head_tl[0])
            tc0 = int(dest["tl"][1] - head_tl[1])
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
                best_dest = dest
        return best_dest, best_dist

    def pick_head_for_container(box_: tuple[int, int, int, int]):
        r0, c0, r1, c1 = box_
        seed_by_color: dict[int, set[tuple[int, int]]] = defaultdict(set)
        for r, c in rect_border_coords(r0, c0, r1, c1):
            for dr, dc in DIR4:
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W:
                    val = int(g[rr, cc])
                    if val not in (int(bg), int(BORDER), int(GREY), int(ORANGE)):
                        seed_by_color[val].add((int(rr), int(cc)))

        best = None
        best_key = None
        for color, seed_pts in seed_by_color.items():
            comp0 = None
            for comp in comp_by_color.get(int(color), []):
                if comp["pts"] & seed_pts:
                    comp0 = comp
                    break
            if comp0 is None:
                continue

            split = split_head_from_component(box_, set(comp0["pts"]))
            if split is None:
                continue

            best_dest, best_dist = find_best_dest(
                box_,
                int(color),
                set(comp0["pts"]),
                set(split["head_pts"]),
                split["head_shape"],
                split["head_tl"],
                split["dir"],
            )
            if best_dest is None:
                continue

            # Prefer the color with the strongest border adjacency signal, then the nearest legal
            # same-colored destination. This is stricter than the original task and guards the
            # multi-color variant against accidental foreign-marker touches.
            key = (len(seed_pts), -int(best_dist))
            if best_key is None or key > best_key:
                best_key = key
                best = {
                    "color": int(color),
                    "head_pts": set(split["head_pts"]),
                    "head_shape": split["head_shape"],
                    "head_tl": split["head_tl"],
                    "comp_pts": set(comp0["pts"]),
                    "dir": split["dir"],
                    "dest": best_dest,
                }
        return best

    for cont in containers:
        r0, c0, r1, c1 = cont["box"]
        head_info = pick_head_for_container(cont["box"])
        if head_info is None:
            continue

        color = int(head_info["color"])
        head_pts = set(head_info["head_pts"])
        head_tl = head_info["head_tl"]
        best_dest = head_info["dest"]
        direction = head_info["dir"]

        tr0 = int(best_dest["tl"][0] - head_tl[0])
        tc0 = int(best_dest["tl"][1] - head_tl[1])
        steps = abs(tr0) + abs(tc0)

        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        grey_coords = [
            (int(r0 + 1 + rr), int(c0 + 1 + cc))
            for rr in range(inner.shape[0])
            for cc in range(inner.shape[1])
            if int(inner[rr, cc]) == GREY
        ]
        steps = min(int(steps), int(len(grey_coords)))

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
            if int(out[int(r), int(c)]) == int(color):
                out[int(r), int(c)] = int(bg)

        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        for rr, cc in to_fill:
            pr = int(rr - r0)
            pc = int(cc - c0)
            if 0 <= pr < patch.shape[0] and 0 <= pc < patch.shape[1] and int(patch[pr, pc]) == GREY:
                patch[pr, pc] = int(ORANGE)
        paste(out, patch, int(r0 + tr), int(c0 + tc), clip=True)

        for r, c in head_pts:
            rr = int(r + tr)
            cc = int(c + tc)
            if 0 <= rr < H and 0 <= cc < W:
                out[rr, cc] = int(color)

    return out


def _interior_fill_order(inner_h: int, inner_w: int, head_side: int) -> list[tuple[int, int]]:
    """Interior cell order from the head-facing side inward, matching solver sort keys exactly."""
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


def _sample_marker_colors(n_containers: int) -> list[int] | None:
    """Use unique visible colors per container; guarantee at least one non-9 color."""
    if n_containers < 2 or n_containers > len(MARKER_COLORS):
        return None
    colors = random.sample(MARKER_COLORS, n_containers)
    if len(set(colors)) < 2:
        return None
    if all(int(c) == 9 for c in colors):
        return None
    return colors


def _gen_attempt() -> np.ndarray | None:
    n_containers = random.randint(2, 4)
    n_decoys = random.choice([0, 0, 1])
    n_movers = max(1, n_containers - n_decoys)
    n_decoys = n_containers - n_movers

    marker_colors = _sample_marker_colors(n_containers)
    if marker_colors is None:
        return None

    specs = []
    for idx in range(n_containers):
        is_decoy = idx >= n_movers
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
                "marker_color": int(marker_colors[idx]),
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
    for asm in assemblies:
        th, tw = asm["total_h"], asm["total_w"]
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
    for asm, (pr, pc) in zip(assemblies, placements):
        oh, ow = asm["outer_h"], asm["outer_w"]
        hs = asm["head_side"]
        gap = asm["gap"]
        inner_h, inner_w = asm["inner_h"], asm["inner_w"]
        initial_orange = asm["initial_orange"]
        marker_color = int(asm["marker_color"])

        cr = pr + asm["cont_r"]
        cc = pc + asm["cont_c"]

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
            grid[cr - 1, cc : cc + ow] = marker_color
            grid[cr - 2 - gap, cc : cc + ow] = marker_color
        elif hs == 1:
            grid[cr + oh, cc : cc + ow] = marker_color
            grid[cr + oh + 1 + gap, cc : cc + ow] = marker_color
        elif hs == 2:
            grid[cr : cr + oh, cc - 1] = marker_color
            grid[cr : cr + oh, cc - 2 - gap] = marker_color
        else:
            grid[cr : cr + oh, cc + ow] = marker_color
            grid[cr : cr + oh, cc + ow + 1 + gap] = marker_color

    used_marker_colors = sorted(int(c) for c in colors_present(grid, exclude={BG, BORDER, GREY, ORANGE}))
    if len(used_marker_colors) < 2 or all(int(c) == 9 for c in used_marker_colors):
        return None

    try:
        output = main(grid)
    except Exception:
        return None

    expected = grid.copy()
    for asm, (pr, pc) in zip(assemblies, placements):
        oh, ow = asm["outer_h"], asm["outer_w"]
        hs = asm["head_side"]
        steps = asm["steps"]
        inner_h, inner_w = asm["inner_h"], asm["inner_w"]
        initial_orange = asm["initial_orange"]
        marker_color = int(asm["marker_color"])
        dr, dc = SIDE_TO_DIR[hs]

        cr = pr + asm["cont_r"]
        cc = pc + asm["cont_c"]

        if steps == 0:
            continue

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
        for idx_cell in range(initial_orange + steps):
            r, c = fill_order[idx_cell]
            interior[r, c] = ORANGE
        patch = np.full((oh, ow), BORDER, dtype=int)
        patch[1:-1, 1:-1] = interior
        expected[nr : nr + oh, nc : nc + ow] = patch

        if hs == 0:
            expected[nr - 1, nc : nc + ow] = marker_color
        elif hs == 1:
            expected[nr + oh, nc : nc + ow] = marker_color
        elif hs == 2:
            expected[nr : nr + oh, nc - 1] = marker_color
        else:
            expected[nr : nr + oh, nc + ow] = marker_color

    if not np.array_equal(output, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid after 300 attempts")
