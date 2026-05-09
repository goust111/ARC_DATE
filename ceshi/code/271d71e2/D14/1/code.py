# Variant of 271d71e2
# Dimension: D14 — 容器移动后的轨迹痕迹 (trajectory)
# Change: 无轨迹 — 容器移动后原位置变为背景色，路径上不留痕迹... → 原位置留下容器'影子'
#
# Design analysis for the D14 variant
# - Concrete behavior:
#   Input stays the same as the original task. The output changes only when a
#   container actually moves: its old box becomes a hollow grey outline, while
#   the moved real container remains black-bordered with the charged interior
#   and translated maroon head. Old head pixels are still erased.
# - Behavioral trace:
#   Original "move right by 2": clear old box to background -> clear old head
#   -> paste shifted box -> paste shifted head.
#   Variant "move right by 2": clear old box to background -> clear old head
#   -> draw grey outline on old box -> paste shifted box -> paste shifted head.
#   The only new artifact is the source-position shadow.
# - Complexity calibration:
#   The shadow uses one fixed shape (old outer rectangle only) and one fixed
#   color (grey). This avoids branching rules while making the trajectory cue
#   visually obvious. The generator keeps at least two movers when possible so
#   the new phenomenon appears multiple times within a sample.
# - Implicit-assumption audit:
#   Detection/matching code reads only the input, so it does not need to handle
#   shadow rectangles. The original erasure line assumed the source vanished
#   completely; in this variant we must redraw the old outline after clearing.
#   The moved container must still be stamped after the shadow so the real box
#   occludes any overlapping shadow cells.
# - Placement legality and observability:
#   The shadow occupies exactly the original container bbox, so it introduces no
#   new global footprint and does not need extra spacing beyond the original
#   assembly placement. Because the shadow is hollow and headless, humans can
#   distinguish it from the real container and from normal grey interior cells.
# - New solver assumptions:
#   The solver now assumes shadow boxes are output-only artifacts and that the
#   input still contains only black-bordered containers. The generator preserves
#   that input grammar and validates every sampled grid against `main()`.
#
from __future__ import annotations
from collections import Counter, defaultdict
from pathlib import Path
import sys
import numpy as np
from scipy.ndimage import label

ROOT = Path(__file__).resolve().parents[5]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import STRUCT4, bbox_coords, colors_present, components_mask, paste
import random

def main(grid: np.ndarray) -> np.ndarray:
    """
    Move bordered containers toward same-colored destination markers, charging gray cells to orange.

    Input structure:
    - bg = 6 (magenta). Black-bordered (0) rectangles with gray (5) / orange (7) interiors.
    - A head sprite (9) sits adjacent to one container edge; a matching destination sprite lies along the movement axis.

    Rule:
    - Each container translates toward its destination along the head-to-dest axis.
    - Steps = min(head-to-dest distance, number of gray interior cells).
    - Gray cells become orange from the head-facing side inward, one per step.

    Key insight: Gray cell count caps movement — a container stops short if it runs out of gray cells to charge.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    BORDER = 0
    GREY = 5
    FILL = 7
    DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # True background is dominant non-border color (border can dominate in some cases).
    cnt = Counter(map(int, g.flatten()))
    bg = int(max((c for c in cnt.keys() if int(c) != int(BORDER)), key=lambda c: int(cnt[int(c)])))

    def rect_border_coords(r0: int, c0: int, r1: int, c1: int):
        for c in range(c0, c1 + 1):
            yield (r0, c)
            yield (r1, c)
        for r in range(r0 + 1, r1):
            yield (r, c0)
            yield (r, c1)

    def draw_rect_outline(arr: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int):
        arr[r0, c0 : c1 + 1] = int(color)
        arr[r1, c0 : c1 + 1] = int(color)
        arr[r0 : r1 + 1, c0] = int(color)
        arr[r0 : r1 + 1, c1] = int(color)

    # --- detect containers as rectangles enclosing GREY/FILL with BORDER frame ---
    containers = []
    for pts in components_mask((g == GREY) | (g == FILL), connectivity=4):
        if pts.size == 0:
            continue
        ir0, ic0, ir1, ic1 = bbox_coords(pts)
        r0, c0, r1, c1 = int(ir0 - 1), int(ic0 - 1), int(ir1 + 1), int(ic1 + 1)
        if r0 < 0 or c0 < 0 or r1 >= H or c1 >= W:
            continue
        if any(int(g[r, c]) != BORDER for r, c in rect_border_coords(r0, c0, r1, c1)):
            continue
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        if inner.size == 0 or not bool(np.isin(inner, [GREY, FILL]).all()):
            continue
        containers.append({"box": (r0, c0, r1, c1)})

    assert not (not containers)

    # head + destination are in some non-bg, non-border color (not GREY/FILL)
    out = g.copy()

    all_head_colors = [c for c in colors_present(g, exclude={bg, BORDER, GREY, FILL})]

    # Precompute components for each head color.
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

    def pick_head_for_container(box_):
        r0, c0, r1, c1 = box_
        head_seeds = []
        for r, c in rect_border_coords(r0, c0, r1, c1):
            for dr, dc in DIR4:
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W:
                    val = int(g[rr, cc])
                    if val not in (int(bg), int(BORDER), int(GREY), int(FILL)):
                        head_seeds.append((rr, cc, val))
        if not head_seeds:
            return None

        hc = int(Counter(val for _r, _c, val in head_seeds).most_common(1)[0][0])
        seed_pts = {(r, c) for r, c, val in head_seeds if int(val) == int(hc)}
        cands = comp_by_color.get(hc, [])
        comp0 = None
        for comp in cands:
            if comp["pts"] & seed_pts:
                comp0 = comp
                break
        if comp0 is None:
            return None

        # Split head vs destination even if same-color pixels touch:
        # head pixels are those with minimal manhattan distance to this container border.
        border_pts = list(rect_border_coords(r0, c0, r1, c1))
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

        # direction from head bbox relative to container box
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

        # normalized head shape + top-left
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
        box_ = cont["box"]
        r0, c0, r1, c1 = box_

        head_info = pick_head_for_container(box_)
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

        # destination candidates
        dests = []
        for d in comp_by_color.get(hc, []):
            if d["shape"] != head_shape:
                continue
            if d["pts"] == comp_pts:
                continue
            dests.append(d)

        # add split-from-same-component destinations (when dest touches head by color)
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
                # reject if adjacent to BORDER
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

        # translation to overlap head onto destination, capped by number of GREY cells
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

        # choose which greys to fill in order from head side inward
        if direction == (0, 1):
            order = sorted(grey_coords, key=lambda p: (-p[1], p[0]))
        elif direction == (0, -1):
            order = sorted(grey_coords, key=lambda p: (p[1], -p[0]))
        elif direction == (1, 0):
            order = sorted(grey_coords, key=lambda p: (-p[0], -p[1]))
        else:  # (-1,0)
            order = sorted(grey_coords, key=lambda p: (p[0], p[1]))
        to_fill = set(order[:steps])

        # erase original container rect and original head pixels (destination stays)
        out[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
        for r, c in head_pts:
            if int(out[int(r), int(c)]) == int(hc):
                out[int(r), int(c)] = int(bg)

        if steps > 0:
            draw_rect_outline(out, r0, c0, r1, c1, GREY)

        # stamp moved container rectangle (using paste for the bulk placement)
        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        if to_fill:
            for rr, cc in to_fill:
                pr = int(rr - r0)
                pc = int(cc - c0)
                if 0 <= pr < patch.shape[0] and 0 <= pc < patch.shape[1] and int(patch[pr, pc]) == GREY:
                    patch[pr, pc] = int(FILL)
        paste(out, patch, int(r0 + tr), int(c0 + tc), clip=True)

        # stamp moved head pixels
        for r, c in head_pts:
            rrr = int(r + tr)
            ccc = int(c + tc)
            if 0 <= rrr < H and 0 <= ccc < W:
                out[rrr, ccc] = int(hc)

    return out

# --------------- generator (attempt 2, Claude) ---------------

BG = 6
BORDER = 0
GREY = 5
ORANGE = 7
HEAD_COLOR = 9

# Direction mapping: head_side -> (dr, dc) movement direction
# head on top -> container moves up; head on bottom -> moves down, etc.
SIDE_TO_DIR = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}


def _interior_fill_order(inner_h: int, inner_w: int, head_side: int) -> list[tuple[int, int]]:
    """Return interior cell coords (local to inner rect) in fill order from head-facing side inward.
    Matches the solver's grey sort keys exactly."""
    cells = [(r, c) for r in range(inner_h) for c in range(inner_w)]
    if head_side == 3:    # right, direction (0,1): key (-col, row)
        cells.sort(key=lambda p: (-p[1], p[0]))
    elif head_side == 2:  # left, direction (0,-1): key (col, -row)
        cells.sort(key=lambda p: (p[1], -p[0]))
    elif head_side == 1:  # bottom, direction (1,0): key (-row, -col)
        cells.sort(key=lambda p: (-p[0], -p[1]))
    else:                 # top, direction (-1,0): key (row, col)
        cells.sort(key=lambda p: (p[0], p[1]))
    return cells


def _gen_attempt() -> np.ndarray | None:
    n_containers = random.randint(2, 4)
    n_decoys = random.choice([0, 0, 1])  # 0 or 1, slight bias toward 0
    n_movers = n_containers - n_decoys
    min_movers = min(2, n_containers)
    if n_movers < min_movers:
        n_movers = min_movers
        n_decoys = n_containers - n_movers

    # Build container specs
    specs = []
    for i in range(n_containers):
        is_decoy = (i >= n_movers)
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

        # Head length = container edge length along head direction
        if head_side in (0, 1):
            head_len = outer_w
        else:
            head_len = outer_h

        specs.append({
            'inner_h': inner_h, 'inner_w': inner_w,
            'outer_h': outer_h, 'outer_w': outer_w,
            'head_side': head_side, 'gap': gap, 'steps': steps,
            'initial_orange': initial_orange, 'grey': grey,
            'head_len': head_len, 'is_decoy': is_decoy,
        })

    # Compute assembly bounding boxes (in input position)
    # Assembly = dest_line + gap + head_line + container
    assemblies = []
    for s in specs:
        oh, ow = s['outer_h'], s['outer_w']
        hs = s['head_side']
        gap = s['gap']

        if hs == 0:  # top: dest above head above container
            total_h = 1 + gap + 1 + oh  # dest + gap + head + container
            total_w = ow
            cont_r = gap + 2  # container top row within assembly
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

        assemblies.append({**s, 'total_h': total_h, 'total_w': total_w,
                           'cont_r': cont_r, 'cont_c': cont_c})

    # Determine grid size
    total_area = sum(a['total_h'] * a['total_w'] for a in assemblies)
    min_side = max(8, int(total_area ** 0.5) + 2)
    max_side = min(26, min_side + 8)
    max_asm_h = max(a['total_h'] for a in assemblies)
    max_asm_w = max(a['total_w'] for a in assemblies)

    lo_h = max(8, max_asm_h + 2)
    hi_h = max(lo_h, min(26, max_side + 4))
    lo_w = max(8, max_asm_w + 2)
    hi_w = max(lo_w, min(26, max_side + 4))

    if lo_h > 26 or lo_w > 26:
        return None

    grid_h = random.randint(lo_h, min(26, hi_h))
    grid_w = random.randint(lo_w, min(26, hi_w))

    # Place assemblies with 1-cell bg gap between them
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    placements = []

    for a in assemblies:
        th, tw = a['total_h'], a['total_w']
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
                occupied[r0:r0+th, c0:c0+tw] = True
                placements.append((r0, c0))
                placed = True
                break
        if not placed:
            return None

    # Build input grid
    grid = np.full((grid_h, grid_w), BG, dtype=int)

    for a, (pr, pc) in zip(assemblies, placements):
        oh, ow = a['outer_h'], a['outer_w']
        hs = a['head_side']
        gap = a['gap']
        steps = a['steps']
        inner_h, inner_w = a['inner_h'], a['inner_w']
        initial_orange = a['initial_orange']

        # Container top-left in grid coords
        cr = pr + a['cont_r']
        cc = pc + a['cont_c']

        # Draw container border
        grid[cr, cc:cc+ow] = BORDER
        grid[cr+oh-1, cc:cc+ow] = BORDER
        grid[cr:cr+oh, cc] = BORDER
        grid[cr:cr+oh, cc+ow-1] = BORDER

        # Fill interior
        fill_order = _interior_fill_order(inner_h, inner_w, hs)
        interior = np.full((inner_h, inner_w), GREY, dtype=int)
        for idx_cell in range(initial_orange):
            r, c = fill_order[idx_cell]
            interior[r, c] = ORANGE
        grid[cr+1:cr+1+inner_h, cc+1:cc+1+inner_w] = interior

        # Draw head line
        if hs == 0:
            grid[cr-1, cc:cc+ow] = HEAD_COLOR
        elif hs == 1:
            grid[cr+oh, cc:cc+ow] = HEAD_COLOR
        elif hs == 2:
            grid[cr:cr+oh, cc-1] = HEAD_COLOR
        else:
            grid[cr:cr+oh, cc+ow] = HEAD_COLOR

        # Draw dest line
        if hs == 0:
            grid[cr-2-gap, cc:cc+ow] = HEAD_COLOR
        elif hs == 1:
            grid[cr+oh+1+gap, cc:cc+ow] = HEAD_COLOR
        elif hs == 2:
            grid[cr:cr+oh, cc-2-gap] = HEAD_COLOR
        else:
            grid[cr:cr+oh, cc+ow+1+gap] = HEAD_COLOR

    # Validate with solver
    try:
        output = main(grid)
    except Exception:
        return None

    # Build expected output manually and compare
    expected = grid.copy()
    for a, (pr, pc) in zip(assemblies, placements):
        oh, ow = a['outer_h'], a['outer_w']
        hs = a['head_side']
        steps = a['steps']
        inner_h, inner_w = a['inner_h'], a['inner_w']
        initial_orange = a['initial_orange']
        dr, dc = SIDE_TO_DIR[hs]

        cr = pr + a['cont_r']
        cc = pc + a['cont_c']

        if steps == 0:
            continue

        # Erase original container + head, then leave the grey source shadow.
        expected[cr:cr+oh, cc:cc+ow] = BG
        if hs == 0:
            expected[cr-1, cc:cc+ow] = BG
        elif hs == 1:
            expected[cr+oh, cc:cc+ow] = BG
        elif hs == 2:
            expected[cr:cr+oh, cc-1] = BG
        else:
            expected[cr:cr+oh, cc+ow] = BG
        expected[cr, cc:cc+ow] = GREY
        expected[cr+oh-1, cc:cc+ow] = GREY
        expected[cr:cr+oh, cc] = GREY
        expected[cr:cr+oh, cc+ow-1] = GREY

        # New position
        nr = cr + dr * steps
        nc = cc + dc * steps

        # Build moved container patch with converted cells
        fill_order = _interior_fill_order(inner_h, inner_w, hs)
        interior = np.full((inner_h, inner_w), GREY, dtype=int)
        for idx_cell in range(initial_orange + steps):
            r, c = fill_order[idx_cell]
            interior[r, c] = ORANGE
        patch = np.full((oh, ow), BORDER, dtype=int)
        patch[1:-1, 1:-1] = interior
        expected[nr:nr+oh, nc:nc+ow] = patch

        # Move head
        if hs == 0:
            expected[nr-1, nc:nc+ow] = HEAD_COLOR
        elif hs == 1:
            expected[nr+oh, nc:nc+ow] = HEAD_COLOR
        elif hs == 2:
            expected[nr:nr+oh, nc-1] = HEAD_COLOR
        else:
            expected[nr:nr+oh, nc+ow] = HEAD_COLOR

    if not np.array_equal(output, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid after 300 attempts")
