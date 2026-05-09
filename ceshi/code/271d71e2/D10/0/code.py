# Variant of 271d71e2
# Dimension: D10 — 容器内部分区结构 (meta)
# Change: 内部分为连续的橙色区域（头侧）和灰色区域（远侧），二者之间是平直分界线... → 灰色和橙色交替条纹

"""
Design analysis for the D10 variant
==================================

a) Concretization and behavior trace
------------------------------------
- Objects kept from the base task:
  - background = 6
  - black rectangular containers with a 1-cell border of 0
  - one 9-colored head line attached to one border side
  - one matching 9-colored destination line on the same axis
- Core D10 change:
  - The interior is no longer "orange prefix + grey suffix".
  - Instead, it is divided into 1-cell-wide stripes orthogonal to motion:
    - head on top/bottom -> horizontal stripes by row
    - head on left/right -> vertical stripes by column
  - Stripe colors alternate from the head side inward.
  - To make "grey stripes from the head side" unambiguous, the first stripe
    at the head side is always grey, then orange, then grey, and so on.
- Motion rule in the variant:
  - head-to-dest distance is unchanged: gap + 1
  - step cap changes from "number of grey cells" to "number of grey stripes"
  - each step converts one whole grey stripe, starting from the head side

  Small trace:
  - Original interior along the motion axis: [orange, orange, grey, grey, grey]
    with width 4 means one step recolors only one grey cell in the nearest grey
    row/column order.
  - Variant interior along the motion axis: [grey, orange, grey, orange, grey]
    with width 4 means one step recolors an entire 4-cell grey stripe.
  - Therefore the same translation distance now corresponds to stripe-count
    rather than pixel-count, and the visible recoloring front is much coarser.

b) Complexity calibration
-------------------------
- If stripes were only 1 pixel total, the variant could degenerate to the
  original behavior. The generator therefore keeps the stripe thickness at 1
  cell but enforces stripe length >= 2 in the orthogonal direction.
- If the stripe count along the motion axis were only 1-2, the alternation
  would be visually weak. The generator enforces axis depth >= 3 for movers.
- The rule uses only one stripe family per container, aligned to the motion
  axis, so the freedom stays small and repeated across examples.

c) Implicit-assumption audit of the original solver
---------------------------------------------------
- Container detection over `(GREY|ORANGE)` as one mask still works because the
  entire interior remains a solid rectangle even though colors alternate.
- Head/destination detection and motion-axis inference do not depend on the
  old contiguous two-block interior, so they stay unchanged.
- The old lines that break are exactly the ones that:
  - count grey *cells* to cap motion
  - sort grey *cells* individually from the head side inward
  - recolor only the first `steps` cells of that order
- In the variant these assumptions are false: a step targets a whole stripe,
  not one cell. The replacement logic groups cells by stripe layer first, then
  counts grey layers and recolors full layers.

d) Placement legality
---------------------
- The base assembly layout is preserved:
  `destination + gap + head + container` along the movement axis.
- Since `steps <= gap + 1`, the moved container and head stay inside the same
  assembly bounding box, so non-overlap of assembly boxes still guarantees a
  legal placement after movement.
- Grid size remains within 8..26 and uses retries if random packing fails.

e) Visual observability
-----------------------
- Every moving container is generated with:
  - at least 3 stripes along the motion axis
  - stripe length at least 2
- This makes the alternating input visible and also makes one-step recoloring
  visibly different from the original per-cell charging rule.
- Fully-orange decoys are removed because they suppress the D10 signal and are
  not needed for the underlying movement relation.

f) Visual neatness
------------------
- Only one border, one head line, one destination line, and one stripe family
  appear per object.
- Containers are separated by at least one background cell between assemblies,
  so the stripes stay readable and do not visually fuse across objects.

g) Color-change inferability
----------------------------
- Only the original task colors are used: 0, 5, 6, 7, 9.
- Recoloring is structured by stripe position relative to the head, not by an
  invisible numeric property of the colors.
- Across samples, the human-visible cue is stable:
  "nearest grey stripe to the head turns orange each time the box advances."

New solver assumptions introduced by this variant
=================================================
- The interior stripes are aligned to the motion axis and span the full width
  or height of the inner rectangle.
- Grey stripes appear in head-to-far order and are the only stripes eligible
  for conversion.
- The generator guarantees these assumptions by construction for every mover.
"""

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

# head on top -> container moves up; head on bottom -> moves down, etc.
SIDE_TO_DIR = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}


def _stripe_layers(inner_h: int, inner_w: int, head_side: int | None = None,
                   direction: tuple[int, int] | None = None) -> list[list[tuple[int, int]]]:
    """Return interior stripes in head-to-far order, using local inner coords."""
    if direction is None:
        assert head_side is not None
        direction = SIDE_TO_DIR[int(head_side)]
    dr, dc = direction
    if (dr, dc) == (-1, 0):
        return [[(r, c) for c in range(inner_w)] for r in range(inner_h)]
    if (dr, dc) == (1, 0):
        return [[(r, c) for c in range(inner_w)] for r in range(inner_h - 1, -1, -1)]
    if (dr, dc) == (0, -1):
        return [[(r, c) for r in range(inner_h)] for c in range(inner_w)]
    if (dr, dc) == (0, 1):
        return [[(r, c) for r in range(inner_h)] for c in range(inner_w - 1, -1, -1)]
    raise ValueError(f"Unsupported direction: {direction}")


def _build_striped_interior(inner_h: int, inner_w: int, head_side: int,
                            converted_grey_stripes: int = 0) -> np.ndarray:
    """
    Build 1-cell stripes from the head side inward.

    Base pattern starts GREY at the head side, then alternates GREY/ORANGE.
    `converted_grey_stripes` turns the first k grey stripes into ORANGE.
    """
    interior = np.full((inner_h, inner_w), ORANGE, dtype=int)
    layers = _stripe_layers(inner_h, inner_w, head_side=head_side)
    grey_layers = layers[::2]
    for layer in grey_layers:
        for r, c in layer:
            interior[r, c] = GREY
    for layer in grey_layers[:converted_grey_stripes]:
        for r, c in layer:
            interior[r, c] = ORANGE
    return interior


def _grey_stripe_count(inner_h: int, inner_w: int, head_side: int) -> int:
    return len(_stripe_layers(inner_h, inner_w, head_side=head_side)[::2])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move bordered containers toward same-colored destination markers.

    Variant rule:
    - Interior colors alternate in stripes from the head side inward.
    - Steps = min(head-to-destination distance, number of grey stripes).
    - Each step converts one entire grey stripe, starting from the head side.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

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

    # Detect containers as rectangles enclosing GREY/ORANGE with a black frame.
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

    # Head + destination are in some non-bg, non-border color (not GREY/ORANGE).
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
        for r, c in rect_border_coords(r0, c0, r1, c1):
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

        # Split head vs destination even if same-color pixels touch:
        # head pixels are those with minimal Manhattan distance to this border.
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

        # Add split-from-same-component destinations when head/dest touch by color.
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
        layers = _stripe_layers(inner.shape[0], inner.shape[1], direction=direction)
        grey_layers = [layer for layer in layers if any(int(inner[rr, cc]) == GREY for rr, cc in layer)]
        steps = min(int(steps), int(len(grey_layers)))

        dr, dc = direction
        tr = int(dr * steps)
        tc = int(dc * steps)
        to_fill_layers = grey_layers[:steps]

        out[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
        for r, c in head_pts:
            if int(out[int(r), int(c)]) == int(hc):
                out[int(r), int(c)] = int(bg)

        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        for layer in to_fill_layers:
            for rr, cc in layer:
                pr = int(rr + 1)
                pc = int(cc + 1)
                if int(patch[pr, pc]) == GREY:
                    patch[pr, pc] = int(ORANGE)
        paste(out, patch, int(r0 + tr), int(c0 + tc), clip=True)

        for r, c in head_pts:
            rrr = int(r + tr)
            ccc = int(c + tc)
            if 0 <= rrr < H and 0 <= ccc < W:
                out[rrr, ccc] = int(hc)

    return out


def _choose_inner_dims(head_side: int) -> tuple[int, int]:
    """
    Pick inner dimensions that make the stripe structure visible.

    Requirements:
    - at least 3 stripes along the motion axis
    - at least 2 cells per stripe so one-step stripe recoloring differs from
      the original per-cell behavior on the same input
    """
    if head_side in (0, 1):
        return random.randint(3, 8), random.randint(2, 5)
    return random.randint(2, 8), random.randint(3, 5)


def _gen_attempt() -> np.ndarray | None:
    n_containers = random.randint(2, 4)

    specs = []
    for _ in range(n_containers):
        head_side = random.randint(0, 3)
        inner_h, inner_w = _choose_inner_dims(head_side)
        outer_h = inner_h + 2
        outer_w = inner_w + 2
        grey_stripes = _grey_stripe_count(inner_h, inner_w, head_side)
        gap = random.randint(0, 10)
        head_to_dest = gap + 1
        steps = min(grey_stripes, head_to_dest)

        specs.append({
            "inner_h": inner_h,
            "inner_w": inner_w,
            "outer_h": outer_h,
            "outer_w": outer_w,
            "head_side": head_side,
            "gap": gap,
            "steps": steps,
            "grey_stripes": grey_stripes,
        })

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
                occupied[r0:r0 + th, c0:c0 + tw] = True
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

        grid[cr, cc:cc + ow] = BORDER
        grid[cr + oh - 1, cc:cc + ow] = BORDER
        grid[cr:cr + oh, cc] = BORDER
        grid[cr:cr + oh, cc + ow - 1] = BORDER
        grid[cr + 1:cr + 1 + inner_h, cc + 1:cc + 1 + inner_w] = _build_striped_interior(
            inner_h, inner_w, hs, converted_grey_stripes=0
        )

        if hs == 0:
            grid[cr - 1, cc:cc + ow] = HEAD_COLOR
            grid[cr - 2 - a["gap"], cc:cc + ow] = HEAD_COLOR
        elif hs == 1:
            grid[cr + oh, cc:cc + ow] = HEAD_COLOR
            grid[cr + oh + 1 + a["gap"], cc:cc + ow] = HEAD_COLOR
        elif hs == 2:
            grid[cr:cr + oh, cc - 1] = HEAD_COLOR
            grid[cr:cr + oh, cc - 2 - a["gap"]] = HEAD_COLOR
        else:
            grid[cr:cr + oh, cc + ow] = HEAD_COLOR
            grid[cr:cr + oh, cc + ow + 1 + a["gap"]] = HEAD_COLOR

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

        expected[cr:cr + oh, cc:cc + ow] = BG
        if hs == 0:
            expected[cr - 1, cc:cc + ow] = BG
        elif hs == 1:
            expected[cr + oh, cc:cc + ow] = BG
        elif hs == 2:
            expected[cr:cr + oh, cc - 1] = BG
        else:
            expected[cr:cr + oh, cc + ow] = BG

        nr = cr + dr * steps
        nc = cc + dc * steps
        patch = np.full((oh, ow), BORDER, dtype=int)
        patch[1:-1, 1:-1] = _build_striped_interior(
            inner_h, inner_w, hs, converted_grey_stripes=steps
        )
        expected[nr:nr + oh, nc:nc + ow] = patch

        if hs == 0:
            expected[nr - 1, nc:nc + ow] = HEAD_COLOR
        elif hs == 1:
            expected[nr + oh, nc:nc + ow] = HEAD_COLOR
        elif hs == 2:
            expected[nr:nr + oh, nc - 1] = HEAD_COLOR
        else:
            expected[nr:nr + oh, nc + ow] = HEAD_COLOR

    if not np.array_equal(output, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid after 300 attempts")
