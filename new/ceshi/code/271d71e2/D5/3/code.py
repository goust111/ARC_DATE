# Variant of 271d71e2
# Dimension: D5 — 灰色像素着色顺序 (multi_step)
# Change: ordered_processing — 灰色像素按从头侧向内的行/列顺序逐个变为橙色... → 螺旋顺序填充

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, bbox_coords, colors_present, components_mask, paste

# Design analysis for this variant:
# - Source-task check: the prompt names `da515329` as the spiral reference. I searched
#   the provided workspace for its solver source, but only the JSON examples are present,
#   so the spiral mechanism below is derived from the visible examples plus the variant brief.
# - Concrete behavior:
#   The black container, head line, destination line, movement axis, and step count all stay
#   exactly as in 271d71e2. The only rule change is which interior grey cells become orange.
#   Instead of "from the head-facing side inward", cells are colored in a fixed clockwise spiral
#   over the interior rectangle: top row left->right, right column top->bottom, bottom row
#   right->left, left column bottom->top, then recurse inward.
# - Small trace:
#   For a 4x4 interior, original top-headed order started row-major from the top.
#   New order is [(0,0),(0,1),(0,2),(0,3),(1,3),(2,3),(3,3),(3,2),...].
#   If the input already has the first 5 spiral cells orange and the object moves 3 steps,
#   the output adds the next 3 spiral cells. Under the original rule those 3 cells would have
#   come from a head-dependent stripe, so the changed pixels differ visibly.
# - Complexity calibration:
#   To avoid degenerate cases where spiral and stripe order look identical, the generator biases
#   movers toward interior sizes >= 2x2 and enforces at least one clearly non-degenerate mover
#   whose spiral result differs from the original stripe result on the same input.
# - Implicit assumptions reviewed:
#   `main()` still assumes each container interior is a full rectangle of only grey/orange cells,
#   and that head/destination detection is unchanged. Those assumptions remain valid because the
#   variant changes only fill order, not object parsing. The old code's only order-sensitive block
#   was the `sorted(grey_coords, key=...)` section; that is the only solver logic replaced.
# - Placement legality:
#   The original assembly bounding box already contains both the input pose and any translated
#   output pose, because the destination lies inside the same strip. Keeping that layout means
#   no new overlap risk is introduced by the spiral rule.
# - Visual observability:
#   The generator keeps 2-4 containers, preserves repeated head directions, and guarantees at
#   least one mover with a partial spiral front, so a human can infer "orange grows clockwise
#   around the perimeter then inward" from a few examples.
# - New solver assumptions after modification:
#   The solver now assumes "clockwise spiral anchored at the interior top-left" is the global
#   ordering. The generator enforces initial orange cells as a spiral prefix, so solver output is
#   consistent and easy to infer rather than mixing unrelated orange layouts.


def _spiral_fill_order(inner_h: int, inner_w: int) -> list[tuple[int, int]]:
    """Clockwise spiral over a rectangle, starting at the top-left corner."""
    order: list[tuple[int, int]] = []
    top, left = 0, 0
    bottom, right = inner_h - 1, inner_w - 1
    while top <= bottom and left <= right:
        for c in range(left, right + 1):
            order.append((top, c))
        for r in range(top + 1, bottom + 1):
            order.append((r, right))
        if top < bottom:
            for c in range(right - 1, left - 1, -1):
                order.append((bottom, c))
        if left < right:
            for r in range(bottom - 1, top, -1):
                order.append((r, left))
        top += 1
        bottom -= 1
        left += 1
        right -= 1
    return order


def _linear_fill_order(inner_h: int, inner_w: int, head_side: int) -> list[tuple[int, int]]:
    """Original 271d71e2 fill order, kept only for generator anti-degeneration checks."""
    cells = [(r, c) for r in range(inner_h) for c in range(inner_w)]
    if head_side == 3:    # right
        cells.sort(key=lambda p: (-p[1], p[0]))
    elif head_side == 2:  # left
        cells.sort(key=lambda p: (p[1], -p[0]))
    elif head_side == 1:  # bottom
        cells.sort(key=lambda p: (-p[0], -p[1]))
    else:                 # top
        cells.sort(key=lambda p: (p[0], p[1]))
    return cells


def _advance_fill(
    initial_orange: set[tuple[int, int]],
    order: list[tuple[int, int]],
    steps: int,
) -> set[tuple[int, int]]:
    out = set(initial_orange)
    added = 0
    for cell in order:
        if cell in out:
            continue
        out.add(cell)
        added += 1
        if added == steps:
            break
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move bordered containers toward same-colored destination markers, charging grey cells to orange.

    Variant change:
    - Translation and step count are unchanged from 271d71e2.
    - Grey cells no longer fill from the head-facing side; they fill in a clockwise spiral
      over the interior rectangle, from the outer ring toward the center.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    BORDER = 0
    GREY = 5
    FILL = 7
    DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    cnt = Counter(map(int, g.flatten()))
    bg = int(max((c for c in cnt.keys() if int(c) != int(BORDER)), key=lambda c: int(cnt[int(c)])))

    def rect_border_coords(r0: int, c0: int, r1: int, c1: int):
        for c in range(c0, c1 + 1):
            yield (r0, c)
            yield (r1, c)
        for r in range(r0 + 1, r1):
            yield (r, c0)
            yield (r, c1)

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

    assert containers

    out = g.copy()
    all_head_colors = [c for c in colors_present(g, exclude={bg, BORDER, GREY, FILL})]

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
        steps = abs(tr0) + abs(tc0)

        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        spiral_order = _spiral_fill_order(inner.shape[0], inner.shape[1])
        spiral_index = {cell: idx for idx, cell in enumerate(spiral_order)}
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

        order = sorted(
            grey_coords,
            key=lambda p: spiral_index[(int(p[0] - (r0 + 1)), int(p[1] - (c0 + 1)))],
        )
        to_fill = set(order[:steps])

        out[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
        for r, c in head_pts:
            if int(out[int(r), int(c)]) == int(hc):
                out[int(r), int(c)] = int(bg)

        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        for rr, cc in to_fill:
            pr = int(rr - r0)
            pc = int(cc - c0)
            if 0 <= pr < patch.shape[0] and 0 <= pc < patch.shape[1] and int(patch[pr, pc]) == GREY:
                patch[pr, pc] = int(FILL)
        paste(out, patch, int(r0 + tr), int(c0 + tc), clip=True)

        for r, c in head_pts:
            rrr = int(r + tr)
            ccc = int(c + tc)
            if 0 <= rrr < H and 0 <= ccc < W:
                out[rrr, ccc] = int(hc)

    return out


BG = 6
BORDER = 0
GREY = 5
ORANGE = 7
HEAD_COLOR = 9

SIDE_TO_DIR = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}


def _spiral_signal_strength(specs: list[dict]) -> bool:
    """Reject samples whose visible behavior would often collapse back to the original stripe rule."""
    clear_mover_found = False
    differs_from_original = False
    for s in specs:
        if s["is_decoy"] or s["steps"] == 0:
            continue
        inner_h = s["inner_h"]
        inner_w = s["inner_w"]
        steps = s["steps"]
        initial_orange_count = s["initial_orange"]
        spiral = _spiral_fill_order(inner_h, inner_w)
        linear = _linear_fill_order(inner_h, inner_w, s["head_side"])
        initial = set(spiral[:initial_orange_count])
        spiral_final = _advance_fill(initial, spiral, steps)
        linear_final = _advance_fill(initial, linear, steps)
        if spiral_final != linear_final:
            differs_from_original = True
        if (
            inner_h >= 3
            and inner_w >= 3
            and steps >= 3
            and len(spiral_final) < inner_h * inner_w
            and spiral_final != linear_final
        ):
            clear_mover_found = True
    return clear_mover_found and differs_from_original


def _gen_attempt() -> np.ndarray | None:
    n_containers = random.randint(2, 4)
    n_decoys = random.choice([0, 0, 1])
    n_movers = max(1, n_containers - n_decoys)
    n_decoys = n_containers - n_movers

    specs = []
    for i in range(n_containers):
        is_decoy = i >= n_movers
        head_side = random.randint(0, 3)

        if is_decoy:
            inner_h = random.randint(1, 6)
            inner_w = random.randint(1, 5)
        else:
            inner_h = random.randint(2, 8)
            inner_w = random.randint(2, 5)

        inner_area = inner_h * inner_w
        outer_h = inner_h + 2
        outer_w = inner_w + 2

        if is_decoy:
            initial_orange = inner_area
            grey = 0
            gap = random.randint(0, 6)
            steps = 0
            cap_mode = "decoy"
        else:
            max_steps = min(7, inner_area - 1)
            if max_steps < 1:
                return None
            mode = random.choice(["distance", "distance", "grey"])
            if mode == "distance":
                lo_steps = 2 if max_steps >= 2 else 1
                steps = random.randint(lo_steps, max_steps)
                grey = random.randint(steps + 1, inner_area)
                initial_orange = inner_area - grey
                gap = steps - 1
            else:
                lo_steps = 1
                hi_steps = min(4, max_steps)
                steps = random.randint(lo_steps, hi_steps)
                grey = steps
                initial_orange = inner_area - grey
                h2d = random.randint(steps + 1, min(10, steps + 4))
                gap = h2d - 1
            cap_mode = mode

        specs.append({
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
            "cap_mode": cap_mode,
        })

    if not _spiral_signal_strength(specs):
        return None

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

        assemblies.append({
            **s,
            "total_h": total_h,
            "total_w": total_w,
            "cont_r": cont_r,
            "cont_c": cont_c,
        })

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

        cr = pr + a["cont_r"]
        cc = pc + a["cont_c"]

        grid[cr, cc : cc + ow] = BORDER
        grid[cr + oh - 1, cc : cc + ow] = BORDER
        grid[cr : cr + oh, cc] = BORDER
        grid[cr : cr + oh, cc + ow - 1] = BORDER

        fill_order = _spiral_fill_order(inner_h, inner_w)
        interior = np.full((inner_h, inner_w), GREY, dtype=int)
        for idx_cell in range(a["initial_orange"]):
            r, c = fill_order[idx_cell]
            interior[r, c] = ORANGE
        grid[cr + 1 : cr + 1 + inner_h, cc + 1 : cc + 1 + inner_w] = interior

        if hs == 0:
            grid[cr - 1, cc : cc + ow] = HEAD_COLOR
            grid[cr - 2 - a["gap"], cc : cc + ow] = HEAD_COLOR
        elif hs == 1:
            grid[cr + oh, cc : cc + ow] = HEAD_COLOR
            grid[cr + oh + 1 + a["gap"], cc : cc + ow] = HEAD_COLOR
        elif hs == 2:
            grid[cr : cr + oh, cc - 1] = HEAD_COLOR
            grid[cr : cr + oh, cc - 2 - a["gap"]] = HEAD_COLOR
        else:
            grid[cr : cr + oh, cc + ow] = HEAD_COLOR
            grid[cr : cr + oh, cc + ow + 1 + a["gap"]] = HEAD_COLOR

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
        dr, dc = SIDE_TO_DIR[hs]

        cr = pr + a["cont_r"]
        cc = pc + a["cont_c"]
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

        fill_order = _spiral_fill_order(inner_h, inner_w)
        interior = np.full((inner_h, inner_w), GREY, dtype=int)
        for idx_cell in range(a["initial_orange"] + steps):
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

    if not np.array_equal(output, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid spiral variant after 500 attempts")
