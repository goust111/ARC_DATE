# Combo variant of 271d71e2 (index 15)
# combo_spec: D5/0+D8/3
# Name: 中心扩散着色加颜色转移

from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, bbox_coords, colors_present, components_mask, paste


# -----------------------------------------------------------------------------
# Design analysis for the combo variant
#
# a) Concretization and behavior trace
#    Source task:
#    - A black-bordered container with grey/orange interior sits next to a colored
#      head line. A same-colored destination line lies farther along the movement axis.
#    - The container moves toward the destination by min(head->dest distance, grey count).
#    - Newly charged cells turn orange from the head-facing side inward.
#    - The destination line remains visible.
#
#    Combo task:
#    - Geometry is unchanged: background, rectangular container, adjacent head line,
#      aligned destination line.
#    - Movement amount is still min(head->dest distance, grey count).
#    - Changed part 1: newly charged cells are chosen from the interior center outward
#      by deterministic 4-neighbor BFS. Existing orange cells are always a prefix of that
#      same center-out order, so each step extends one visible center-grown blob.
#    - Changed part 2: when a container moves at least one cell, the destination line is
#      consumed. It disappears to background, and its color is copied onto the moved
#      container border. The moved head keeps the same line color.
#
#    Small trace:
#    - Suppose a 5x3 interior has 6 orange cells already. In the source rule with head
#      above, the next 2 charged cells would come from the top side. In this combo, the
#      next 2 come from the nearest remaining cells to the center seed set.
#    - After movement, the source task would leave a black border and keep the target
#      line. The combo instead erases the target line and recolors the moved border to
#      the target/head color.
#
# b) Complexity calibration
#    - To keep the center-out pattern human-visible, mover interiors are biased toward
#      odd 2D sizes such as 3x3, 5x3, 5x5, 7x5. These have a visually stable center.
#    - To keep color transfer inferable, line colors come from a small palette
#      {1,2,3,4,8,9} and are unique per container within each grid, avoiding ambiguous
#      head/destination matching.
#    - Every generated grid contains at least two informative movers:
#      1. a distance-limited showcase mover, so center-out charging differs from the
#         original head-side charging in the final visible orange pattern,
#      2. a grey-limited mover, so the destination disappearance leaves a visible gap.
#
# c) Hidden-assumption audit of the original main()
#    - Container detection assumes input borders are black. Still valid: only outputs
#      recolor borders; generated inputs keep black borders.
#    - Head/destination detection assumes both are in some non-bg/non-border/non-5/non-7
#      color. Still valid: line colors are drawn from a small non-overlapping palette.
#    - Destination matching assumes head and destination share color and shape. Still
#      valid; generation preserves that and makes colors unique per assembly.
#    - Original fill logic assumed direction determined charge order. No longer valid.
#      Direction still determines translation, but charging order now depends only on the
#      geometric center BFS.
#    - Original erase logic preserved the destination. No longer valid. We must explicitly
#      erase the chosen destination before stamping the moved object.
#    - Original patch stamping preserved black borders. No longer valid. We must recolor
#      border pixels in the moved patch to the destination/head color.
#    - Original zero-step behavior could be handled by a paste-at-same-place no-op. That
#      would be wrong here because it would recolor a non-moving decoy border. Therefore
#      steps<=0 must short-circuit and preserve the whole assembly unchanged.
#
# d) Placement legality
#    - Each assembly reserves destination + gap + head + container along one axis, exactly
#      as in the source task. Movement never overshoots the destination, so the output
#      remains inside the reserved assembly footprint.
#    - Assemblies are placed with at least a 1-cell background moat, so no overlaps occur
#      in input or output, and erasing a destination cannot damage another object.
#
# e) Visual observability
#    - Humans can see two independent signals:
#      1. orange spreads from the middle rather than from the head side,
#      2. the destination disappears and its color reappears on the moved border.
#    - At least one mover per sample is constructed so the combo output differs from the
#      original task's output on the same input.
#
# f) Visual cleanliness
#    - No extra markers are added. Each assembly still contains only one container, one
#      head line, and one destination line.
#    - Unique line colors and background moats keep assemblies visually separate.
#
# g) Color-change inferability
#    - Only one color rule is added: "the line color becomes the moved border color".
#      This is anchored by shared geometry and repeated across examples, not by hidden
#      numeric ids.
#
# h) Generator-side non-degeneracy checks
#    - Generated inputs are validated against the combo solver.
#    - Generation rejects grids that fail to include a showcase mover whose final orange
#      set differs from the source task's head-side ordering.
#    - Generation also guarantees a grey-limited mover so target disappearance is visible.
#
# i) New assumptions introduced by the combo solver and how generation satisfies them
#    - Assumption: existing orange cells are a prefix of the center-out order, so taking
#      the first remaining grey cells in that order reproduces the intended evolution.
#      Generation enforces exactly that.
#    - Assumption: every mover has one intended same-colored destination candidate on its
#      axis. Generation uses unique per-container line colors, so matching is unambiguous.
#    - Assumption: only movers recolor borders. Generation includes 0-1 fully orange
#      decoys, and main() preserves them by skipping zero-step cases.
# -----------------------------------------------------------------------------


BG = 6
BORDER = 0
GREY = 5
ORANGE = 7
LINE_COLORS = (1, 2, 3, 4, 8, 9)

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDE_TO_DIR = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}


def _rect_border_coords(r0: int, c0: int, r1: int, c1: int):
    for c in range(c0, c1 + 1):
        yield (r0, c)
        yield (r1, c)
    for r in range(r0 + 1, r1):
        yield (r, c0)
        yield (r, c1)


def _paint_line(
    grid: np.ndarray, side: int, cr: int, cc: int, oh: int, ow: int, dist: int, color: int
) -> None:
    if side == 0:
        grid[cr - dist, cc : cc + ow] = int(color)
    elif side == 1:
        grid[cr + oh - 1 + dist, cc : cc + ow] = int(color)
    elif side == 2:
        grid[cr : cr + oh, cc - dist] = int(color)
    else:
        grid[cr : cr + oh, cc + ow - 1 + dist] = int(color)


def _center_seed_cells(inner_h: int, inner_w: int) -> list[tuple[int, int]]:
    target_r2 = inner_h - 1
    target_c2 = inner_w - 1
    best = None
    seeds: list[tuple[int, int]] = []
    for r in range(inner_h):
        for c in range(inner_w):
            dist2 = abs(2 * r - target_r2) + abs(2 * c - target_c2)
            if best is None or dist2 < best:
                best = dist2
                seeds = [(r, c)]
            elif dist2 == best:
                seeds.append((r, c))
    seeds.sort()
    return seeds


def _center_out_order(inner_h: int, inner_w: int) -> list[tuple[int, int]]:
    seeds = _center_seed_cells(inner_h, inner_w)
    q = deque(seeds)
    seen = set(seeds)
    order: list[tuple[int, int]] = []
    nbrs = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    while q:
        r, c = q.popleft()
        order.append((r, c))
        for dr, dc in nbrs:
            rr, cc = r + dr, c + dc
            if 0 <= rr < inner_h and 0 <= cc < inner_w and (rr, cc) not in seen:
                seen.add((rr, cc))
                q.append((rr, cc))
    return order


def _head_side_order(inner_h: int, inner_w: int, head_side: int) -> list[tuple[int, int]]:
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


def _old_rule_final_orange_set(
    inner_h: int, inner_w: int, head_side: int, initial_orange: int, steps: int
) -> set[tuple[int, int]]:
    orange_now = set(_center_out_order(inner_h, inner_w)[:initial_orange])
    old_fill: list[tuple[int, int]] = []
    for cell in _head_side_order(inner_h, inner_w, head_side):
        if cell in orange_now:
            continue
        old_fill.append(cell)
        if len(old_fill) == steps:
            break
    return orange_now | set(old_fill)


def _container_components(grid: np.ndarray) -> list[dict]:
    g = np.array(grid, dtype=int)
    h, w = g.shape
    containers = []
    for pts in components_mask((g == GREY) | (g == ORANGE), connectivity=4):
        if pts.size == 0:
            continue
        ir0, ic0, ir1, ic1 = bbox_coords(pts)
        r0, c0, r1, c1 = int(ir0 - 1), int(ic0 - 1), int(ir1 + 1), int(ic1 + 1)
        if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
            continue
        if any(int(g[r, c]) != int(BORDER) for r, c in _rect_border_coords(r0, c0, r1, c1)):
            continue
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        if inner.size == 0 or not bool(np.isin(inner, [GREY, ORANGE]).all()):
            continue
        containers.append({"box": (r0, c0, r1, c1)})
    return containers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move bordered containers toward same-colored destination lines.

    Combo rule:
    - steps = min(head->dest distance, remaining grey count)
    - newly charged grey cells turn orange in center-out BFS order
    - when steps > 0, the destination line disappears and its color is transferred
      to the moved container border
    """
    g = np.array(grid, dtype=int)
    h, w = g.shape

    cnt = Counter(map(int, g.flatten()))
    bg = int(max((c for c in cnt.keys() if int(c) != int(BORDER)), key=lambda c: int(cnt[int(c)])))

    containers = _container_components(g)
    assert containers

    out = g.copy()
    all_line_colors = [c for c in colors_present(g, exclude={bg, BORDER, GREY, ORANGE})]

    comp_by_color: dict[int, list[dict]] = defaultdict(list)
    for col in all_line_colors:
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
                    if val not in (int(bg), int(BORDER), int(GREY), int(ORANGE)):
                        head_seeds.append((rr, cc, val))
        if not head_seeds:
            return None

        line_color = int(Counter(val for _r, _c, val in head_seeds).most_common(1)[0][0])
        seed_pts = {(r, c) for r, c, val in head_seeds if int(val) == int(line_color)}
        comp0 = None
        for comp in comp_by_color.get(line_color, []):
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
            "color": line_color,
            "head_pts": head_pts,
            "head_shape": head_shape,
            "head_tl": (r00, c00),
            "comp_pts": set(comp0["pts"]),
            "dir": direction,
        }

    for cont in containers:
        r0, c0, r1, c1 = cont["box"]
        head_info = pick_head_for_container((r0, c0, r1, c1))
        if head_info is None:
            continue

        line_color = int(head_info["color"])
        head_pts = set(head_info["head_pts"])
        head_shape = head_info["head_shape"]
        head_tl = head_info["head_tl"]
        comp_pts = set(head_info["comp_pts"])
        direction = head_info["dir"]
        if direction == (0, 0):
            continue

        dests = []
        for d in comp_by_color.get(line_color, []):
            if d["shape"] != head_shape or d["pts"] == comp_pts:
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
                        if 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == int(BORDER):
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
        steps = abs(tr0) + abs(tc0)

        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        center_order = _center_out_order(inner.shape[0], inner.shape[1])
        remaining_greys = [(rr, cc) for rr, cc in center_order if int(inner[rr, cc]) == int(GREY)]
        steps = min(int(steps), int(len(remaining_greys)))
        if steps <= 0:
            continue

        dr, dc = direction
        tr = int(dr * steps)
        tc = int(dc * steps)

        out[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
        for r, c in head_pts:
            if int(out[int(r), int(c)]) == int(line_color):
                out[int(r), int(c)] = int(bg)
        for r, c in best_dest["pts"]:
            if int(out[int(r), int(c)]) == int(line_color):
                out[int(r), int(c)] = int(bg)

        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        for rr, cc in remaining_greys[:steps]:
            if int(patch[rr + 1, cc + 1]) == int(GREY):
                patch[rr + 1, cc + 1] = int(ORANGE)
        patch[patch == int(BORDER)] = int(line_color)
        paste(out, patch, int(r0 + tr), int(c0 + tc), clip=True)

        for r, c in head_pts:
            rrr = int(r + tr)
            ccc = int(c + tc)
            if 0 <= rrr < h and 0 <= ccc < w:
                out[rrr, ccc] = int(line_color)

    return out


def _make_showcase_mover() -> dict:
    for _ in range(300):
        head_side = random.randint(0, 3)
        inner_h = random.choice([3, 5, 7])
        inner_w = random.choice([3, 5])
        inner_area = inner_h * inner_w
        h2d = random.randint(2, min(8, inner_area - 1))
        grey = random.randint(h2d + 1, inner_area)
        steps = h2d
        initial_orange = inner_area - grey
        new_final = set(_center_out_order(inner_h, inner_w)[: initial_orange + steps])
        old_final = _old_rule_final_orange_set(inner_h, inner_w, head_side, initial_orange, steps)
        if steps >= 2 and new_final != old_final:
            return {
                "inner_h": inner_h,
                "inner_w": inner_w,
                "head_side": head_side,
                "gap": h2d - 1,
                "steps": steps,
                "initial_orange": initial_orange,
                "grey": grey,
                "is_decoy": False,
                "cap_type": "showcase",
            }
    raise RuntimeError("failed to create showcase mover")


def _make_other_mover(cap_type: str) -> dict:
    for _ in range(300):
        head_side = random.randint(0, 3)
        inner_h = random.choice([3, 5, 7])
        inner_w = random.choice([3, 5])
        inner_area = inner_h * inner_w
        if cap_type == "grey":
            h2d = random.randint(2, 10)
            grey = random.randint(1, min(inner_area, h2d - 1))
            steps = grey
        else:
            h2d = random.randint(1, min(10, inner_area))
            grey = random.randint(h2d, inner_area)
            steps = h2d
        initial_orange = inner_area - grey
        return {
            "inner_h": inner_h,
            "inner_w": inner_w,
            "head_side": head_side,
            "gap": h2d - 1,
            "steps": steps,
            "initial_orange": initial_orange,
            "grey": grey,
            "is_decoy": False,
            "cap_type": cap_type,
        }
    raise RuntimeError("failed to create mover")


def _make_decoy() -> dict:
    inner_h = random.randint(1, 8)
    inner_w = random.randint(1, 5)
    inner_area = inner_h * inner_w
    return {
        "inner_h": inner_h,
        "inner_w": inner_w,
        "head_side": random.randint(0, 3),
        "gap": random.randint(0, 6),
        "steps": 0,
        "initial_orange": inner_area,
        "grey": 0,
        "is_decoy": True,
        "cap_type": "decoy",
    }


def _specs_visible_under_combo(specs: list[dict]) -> tuple[bool, bool]:
    visible_center = False
    visible_transfer = False
    for spec in specs:
        if spec["is_decoy"] or spec["steps"] <= 0:
            continue
        inner_h = int(spec["inner_h"])
        inner_w = int(spec["inner_w"])
        initial_orange = int(spec["initial_orange"])
        steps = int(spec["steps"])
        head_side = int(spec["head_side"])
        new_final = set(_center_out_order(inner_h, inner_w)[: initial_orange + steps])
        old_final = _old_rule_final_orange_set(inner_h, inner_w, head_side, initial_orange, steps)
        if steps >= 2 and new_final != old_final:
            visible_center = True
        if spec["cap_type"] in ("grey", "showcase"):
            visible_transfer = True
    return visible_center, visible_transfer


def _gen_attempt() -> np.ndarray | None:
    n_containers = random.randint(2, 4)
    n_decoys = random.choice([0, 0, 1])
    n_movers = max(2, n_containers - n_decoys)
    n_movers = min(n_movers, n_containers)
    n_decoys = n_containers - n_movers

    mover_specs = [_make_showcase_mover(), _make_other_mover("grey")]
    while len(mover_specs) < n_movers:
        mover_specs.append(_make_other_mover(random.choice(["grey", "distance"])))
    specs = mover_specs + [_make_decoy() for _ in range(n_decoys)]
    random.shuffle(specs)

    visible_center, visible_transfer = _specs_visible_under_combo(specs)
    if not visible_center or not visible_transfer:
        return None

    line_colors = random.sample(LINE_COLORS, k=n_containers)
    for spec, line_color in zip(specs, line_colors):
        spec["line_color"] = int(line_color)
        spec["outer_h"] = int(spec["inner_h"] + 2)
        spec["outer_w"] = int(spec["inner_w"] + 2)

    assemblies = []
    for spec in specs:
        oh = int(spec["outer_h"])
        ow = int(spec["outer_w"])
        hs = int(spec["head_side"])
        gap = int(spec["gap"])

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

        assemblies.append(
            {
                **spec,
                "total_h": total_h,
                "total_w": total_w,
                "cont_r": cont_r,
                "cont_c": cont_c,
            }
        )

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
    for assembly in assemblies:
        th = int(assembly["total_h"])
        tw = int(assembly["total_w"])
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
        oh = int(assembly["outer_h"])
        ow = int(assembly["outer_w"])
        hs = int(assembly["head_side"])
        inner_h = int(assembly["inner_h"])
        inner_w = int(assembly["inner_w"])
        initial_orange = int(assembly["initial_orange"])
        line_color = int(assembly["line_color"])

        cr = int(pr + assembly["cont_r"])
        cc = int(pc + assembly["cont_c"])

        grid[cr, cc : cc + ow] = int(BORDER)
        grid[cr + oh - 1, cc : cc + ow] = int(BORDER)
        grid[cr : cr + oh, cc] = int(BORDER)
        grid[cr : cr + oh, cc + ow - 1] = int(BORDER)

        interior = np.full((inner_h, inner_w), GREY, dtype=int)
        center_order = _center_out_order(inner_h, inner_w)
        for idx_cell in range(initial_orange):
            r, c = center_order[idx_cell]
            interior[r, c] = int(ORANGE)
        grid[cr + 1 : cr + 1 + inner_h, cc + 1 : cc + 1 + inner_w] = interior

        _paint_line(grid, hs, cr, cc, oh, ow, 1, line_color)
        _paint_line(grid, hs, cr, cc, oh, ow, int(assembly["gap"]) + 2, line_color)

    try:
        output = main(grid)
    except Exception:
        return None

    expected = grid.copy()
    for assembly, (pr, pc) in zip(assemblies, placements):
        oh = int(assembly["outer_h"])
        ow = int(assembly["outer_w"])
        hs = int(assembly["head_side"])
        steps = int(assembly["steps"])
        inner_h = int(assembly["inner_h"])
        inner_w = int(assembly["inner_w"])
        initial_orange = int(assembly["initial_orange"])
        line_color = int(assembly["line_color"])
        dr, dc = SIDE_TO_DIR[hs]

        cr = int(pr + assembly["cont_r"])
        cc = int(pc + assembly["cont_c"])

        if steps <= 0:
            continue

        expected[cr : cr + oh, cc : cc + ow] = int(BG)
        _paint_line(expected, hs, cr, cc, oh, ow, 1, BG)
        _paint_line(expected, hs, cr, cc, oh, ow, int(assembly["gap"]) + 2, BG)

        nr = int(cr + dr * steps)
        nc = int(cc + dc * steps)
        center_order = _center_out_order(inner_h, inner_w)

        interior = np.full((inner_h, inner_w), GREY, dtype=int)
        for idx_cell in range(initial_orange + steps):
            r, c = center_order[idx_cell]
            interior[r, c] = int(ORANGE)

        patch = np.full((oh, ow), int(line_color), dtype=int)
        patch[1:-1, 1:-1] = interior
        expected[nr : nr + oh, nc : nc + ow] = patch
        _paint_line(expected, hs, nr, nc, oh, ow, 1, line_color)

    if not np.array_equal(output, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid combo variant after 400 attempts")
