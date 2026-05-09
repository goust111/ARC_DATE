# Variant of 271d71e2
# Dimension: D8 — 变换类型 (transform)
# Change: recolor — 灰色(5)变为橙色(7)，容器整体平移，目标线保持不动
#   → 目标线颜色转移到容器：容器移动后，目标线消失（变为背景），其颜色被转移到容器边框上

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, bbox_coords, colors_present, components_mask, paste


# -----------------------------------------------------------------------------
# Design analysis for the D8 variant
#
# 0) Reference-task note
#    The prompt cites task 4c416de3 for "color_transfer". Within the allowed local
#    files, its evaluation JSON is present but a readable solver source file is not.
#    So this variant only borrows the requested mechanism-level idea:
#    "a marker's visible color is transferred onto another object", while keeping the
#    original 271d71e2 geometry and motion logic almost unchanged.
#
# a) Concretization and behavior trace
#    Input still contains:
#    - magenta background 6
#    - black rectangular container borders 0
#    - gray/orange interior 5/7
#    - a 1-cell-thick head line adjacent to one side of the container
#    - an aligned destination line farther along the same axis
#
#    The key visible change is in the output:
#    - the container still moves by min(head-to-dest distance, gray count)
#    - gray interior cells still turn orange from the head-facing side inward
#    - BUT the destination line no longer remains in place
#    - instead, its line color is copied onto the moved container border
#
#    Concrete trace on a top-headed container:
#    - Original task:
#      input = black box, maroon head above it, maroon destination farther above
#      output = moved black box, more orange inside, moved maroon head, destination
#      still visible if the box stopped short
#    - Variant:
#      input = same geometry, but line color may be blue/red/green/yellow/teal/maroon
#      output = moved box whose border is recolored to that line color, moved head
#      keeps moving, destination line is erased to background
#    - Difference:
#      the "target marker" is consumed and its color becomes part of the moved object.
#
# b) Complexity calibration
#    If line color stayed fixed at 9, the transfer rule would be weaker visually:
#    users would only notice "black border became 9". To make D8 observable without
#    adding unrelated complexity, the generator varies line colors from a small palette
#    {1,2,3,4,8,9}. Colors are unique per container within a grid, avoiding matching
#    ambiguity while making the transfer rule easy to see.
#
#    Another observability risk is degeneration into the original behavior when the head
#    reaches the destination exactly: the destination disappears, but the moved head can
#    land on the same cells, partially hiding that fact. So each generated grid is forced
#    to contain at least one "gray-limited" mover with steps < head-to-dest distance,
#    ensuring a visible empty gap where the destination used to be.
#
# c) Hidden-assumption audit of the original main()
#    - Container detection depends on borders being black in the input. This still holds:
#      only the output border changes color; generated inputs keep black borders.
#    - Head/destination discovery assumes both are in some non-bg/non-border/non-5/non-7
#      color. Still valid because line colors are drawn from {1,2,3,4,8,9}.
#    - Destination matching assumes head and destination share a color and shape.
#      Still valid; only the fixed value 9 is relaxed to a small palette.
#    - Original erase logic only removed the old container and old head, keeping the
#      destination. That is no longer valid. The variant must explicitly erase the chosen
#      destination component before drawing the moved object.
#    - Original patch paste preserved black border pixels. That is no longer valid. The
#      variant must recolor border pixels in the moved patch to the destination color.
#    - Original zero-step path was harmless because paste-at-same-place produced no net
#      change. In this variant, blindly recoloring the border at steps=0 would create a
#      new unintended rule ("fully orange decoys recolor without moving"). Therefore
#      steps==0 must short-circuit and leave the whole assembly unchanged.
#
# d) Placement legality
#    The generator keeps the original axial layout:
#    destination + gap + head + container (or the mirrored order by side).
#    Assemblies are placed with a 1-cell background moat. Because movement is only toward
#    the destination and never beyond it, the output stays inside the same assembly bbox.
#    Removing the destination in the output only reduces occupied area, so no new overlap
#    risk is introduced.
#
# e) Visual observability
#    Human-visible cues per example:
#    - destination line vanishes
#    - moved container border turns into the exact line color
#    - gray-to-orange charging still tracks movement amount
#    Using 2-4 containers per grid with repeated "line color -> border color" behavior
#    gives enough support for the transfer rule from just a few examples.
#
# f) Visual cleanliness
#    The generator keeps only one layer of information around each container:
#    border, interior, head, destination. No extra glyphs are added. Unique line colors
#    per container avoid same-color cross-grid confusion. The 1-cell moat keeps different
#    assemblies visually separated.
#
# g) Color-change inferability
#    The only new color logic is: "the color of the destination/head line becomes the
#    color of the moved border." This is structurally anchored by adjacency and shared
#    movement axis, not by hidden numeric color ids. The palette is small and reused
#    across samples, making the mapping inferable.
#
# 9) New assumptions introduced by the modified main() and how generate_input() satisfies
#    them
#    - Assumption: every moving container has exactly one aligned same-colored destination
#      candidate of matching shape along its movement axis. Generation uses unique colors
#      per container within a grid, so this is guaranteed.
#    - Assumption: the chosen destination can be erased safely without affecting another
#      object. Generation reserves a 1-cell background moat between assemblies, so each
#      destination belongs to only one assembly.
#    - Assumption: at least some examples visibly exercise "destination disappears before
#      the head reaches it". Generation forces >=1 gray-limited mover per grid.
#    - Assumption: zero-step containers should remain unchanged. Generation includes 0-1
#      fully orange decoys, and main() now explicitly preserves them.
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


def _interior_fill_order(inner_h: int, inner_w: int, head_side: int) -> list[tuple[int, int]]:
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


def _paint_line(grid: np.ndarray, side: int, cr: int, cc: int, oh: int, ow: int, dist: int, color: int) -> None:
    if side == 0:
        grid[cr - dist, cc : cc + ow] = int(color)
    elif side == 1:
        grid[cr + oh - 1 + dist, cc : cc + ow] = int(color)
    elif side == 2:
        grid[cr : cr + oh, cc - dist] = int(color)
    else:
        grid[cr : cr + oh, cc + ow - 1 + dist] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move bordered containers toward same-colored destination lines, charging gray cells
    to orange. In this D8 variant, the destination line is consumed: it disappears and
    its color is transferred onto the moved container border.
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
        steps = abs(tr0) + abs(tc0)

        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        grey_coords = [
            (int(r0 + 1 + rr), int(c0 + 1 + cc))
            for rr in range(inner.shape[0])
            for cc in range(inner.shape[1])
            if int(inner[rr, cc]) == int(GREY)
        ]
        steps = min(int(steps), int(len(grey_coords)))
        if steps <= 0:
            continue

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
            if int(out[int(r), int(c)]) == int(line_color):
                out[int(r), int(c)] = int(bg)
        for r, c in best_dest["pts"]:
            if int(out[int(r), int(c)]) == int(line_color):
                out[int(r), int(c)] = int(bg)

        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        for rr, cc in to_fill:
            pr = int(rr - r0)
            pc = int(cc - c0)
            if 0 <= pr < patch.shape[0] and 0 <= pc < patch.shape[1] and int(patch[pr, pc]) == int(GREY):
                patch[pr, pc] = int(ORANGE)
        patch[patch == int(BORDER)] = int(line_color)
        paste(out, patch, int(r0 + tr), int(c0 + tc), clip=True)

        for r, c in head_pts:
            rrr = int(r + tr)
            ccc = int(c + tc)
            if 0 <= rrr < H and 0 <= ccc < W:
                out[rrr, ccc] = int(line_color)

    return out


def _gen_attempt() -> np.ndarray | None:
    n_containers = random.randint(2, 4)
    n_decoys = random.choice([0, 0, 1])
    n_movers = max(1, n_containers - n_decoys)
    n_decoys = n_containers - n_movers

    mover_cap_types = ["grey"]
    while len(mover_cap_types) < n_movers:
        mover_cap_types.append(random.choice(["grey", "distance"]))
    random.shuffle(mover_cap_types)

    line_colors = random.sample(LINE_COLORS, k=n_containers)

    specs = []
    mover_idx = 0
    for i in range(n_containers):
        is_decoy = i >= n_movers
        head_side = random.randint(0, 3)
        inner_h = random.randint(1, 8)
        inner_w = random.randint(1, 5)

        if not is_decoy and mover_cap_types[mover_idx] == "grey" and inner_h * inner_w == 1:
            inner_w = 2

        inner_area = inner_h * inner_w
        outer_h = inner_h + 2
        outer_w = inner_w + 2

        if is_decoy:
            grey = 0
            initial_orange = inner_area
            h2d = random.randint(1, 7)
            steps = 0
            cap_type = "decoy"
        else:
            cap_type = mover_cap_types[mover_idx]
            mover_idx += 1
            if cap_type == "grey":
                h2d = random.randint(2, 10)
                grey = random.randint(1, min(inner_area, h2d - 1))
                steps = grey
            else:
                h2d = random.randint(1, min(10, inner_area))
                grey = random.randint(h2d, inner_area)
                steps = h2d
            initial_orange = inner_area - grey

        gap = h2d - 1
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
                "line_color": line_colors[i],
                "is_decoy": is_decoy,
                "cap_type": cap_type,
            }
        )

    assemblies = []
    for s in specs:
        oh, ow = s["outer_h"], s["outer_w"]
        hs = s["head_side"]
        gap = s["gap"]

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
    for a, (pr, pc) in zip(assemblies, placements):
        oh, ow = a["outer_h"], a["outer_w"]
        hs = a["head_side"]
        inner_h, inner_w = a["inner_h"], a["inner_w"]
        initial_orange = a["initial_orange"]
        line_color = a["line_color"]

        cr = pr + a["cont_r"]
        cc = pc + a["cont_c"]

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

        _paint_line(grid, hs, cr, cc, oh, ow, 1, line_color)
        _paint_line(grid, hs, cr, cc, oh, ow, a["gap"] + 2, line_color)

    try:
        output = main(grid)
    except Exception:
        return None

    expected = grid.copy()
    for a, (pr, pc) in zip(assemblies, placements):
        oh, ow = a["outer_h"], a["outer_w"]
        hs = a["head_side"]
        steps = a["steps"]
        inner_h, inner_w = a["inner_h"], a["inner_w"]
        initial_orange = a["initial_orange"]
        line_color = a["line_color"]
        dr, dc = SIDE_TO_DIR[hs]

        cr = pr + a["cont_r"]
        cc = pc + a["cont_c"]

        if steps <= 0:
            continue

        expected[cr : cr + oh, cc : cc + ow] = int(BG)
        _paint_line(expected, hs, cr, cc, oh, ow, 1, BG)
        _paint_line(expected, hs, cr, cc, oh, ow, a["gap"] + 2, BG)

        nr = cr + dr * steps
        nc = cc + dc * steps

        fill_order = _interior_fill_order(inner_h, inner_w, hs)
        interior = np.full((inner_h, inner_w), GREY, dtype=int)
        for idx_cell in range(initial_orange + steps):
            r, c = fill_order[idx_cell]
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
    raise RuntimeError("Failed to generate valid variant after 400 attempts")
