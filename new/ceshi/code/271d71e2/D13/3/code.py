# Variant of 271d71e2
# Dimension: D13 — 步数计算的数学关系 (meta)
# Change: steps = min(head_dest_distance, grey_count)，其中 head_dest_distance = gap + 1
#   -> steps = grey_count / inner_width（整除）. More precisely, the motion budget is the
#   number of complete interior layers of grey: vertical motion uses grey_count // inner_w,
#   horizontal motion uses grey_count // inner_h.
#
# Design analysis
# a) Concrete behavior:
#    - The visual scene stays the same as the base task: magenta background, black-bordered
#      containers, grey/orange interiors, and maroon head/destination lines.
#    - The changed part is the translation amount. A container no longer cares about the
#      head-to-destination distance, except that the destination still indicates direction.
#      Instead, movement counts complete grey rows/columns.
#    - Example: head on top, inner size 4x3, interior prefix order is row-major from top.
#      If 7 cells are grey, the old rule moves min(distance, 7). The new rule moves 7 // 3 = 2.
#      Two full 3-cell row-equivalents are consumed; one leftover grey cell remains.
# b) Complexity calibration:
#    - To make the variant visible, generated movers use layer spans > 1 and enforce
#      head_dest_distance > new_steps, so old and new behaviors differ.
#    - At least one mover per grid has grey_count % layer_span != 0, exposing the floor rule.
# c) Implicit assumptions review:
#    - Container detection, head detection, destination matching, and motion axis inference
#      are unchanged; they do not depend on the old arithmetic.
#    - The old solver used one `steps` variable both for displacement and for the number of
#      grey cells to recolor. Under the new semantics that is no longer coherent: one step is
#      one complete grey layer, not one pixel. The solver therefore recolors `steps * layer_span`
#      grey cells from the head side inward. This is the smallest adaptation that keeps the
#      D13 change legible in the output.
#    - The recolor logic still assumes the interior follows the original "orange prefix, grey
#      suffix in head-order" structure. The generator preserves that invariant.
# d) Placement legality:
#    - The generator chooses head_dest_distance strictly larger than the computed steps for
#      every mover. Therefore the translated container always remains inside the same
#      destination-to-container assembly rectangle used at input placement time.
#    - Assemblies are placed with a one-cell background moat, so input and output stay
#      non-overlapping and solver matching remains unambiguous.
# e) Visual observability:
#    - Samples show that wide/tall grey regions may move only a few cells because movement
#      depends on complete rows/columns, not raw grey area.
#    - Leftover partial grey layers visibly remain after motion, which exposes the floor.
# f) Visual cleanliness:
#    - No new markers are introduced. Separation remains the same as the base task.
# g) Color inference:
#    - No color semantics change. Only the arithmetic relation between grey structure and
#      movement changes, so the task remains readable from 2-4 examples.

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

# Direction mapping: head on top -> move up, etc.
SIDE_TO_DIR = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

# Exported for post-generation diagnostics.
LAST_GENERATED_META: dict | None = None


def _rect_border_coords(r0: int, c0: int, r1: int, c1: int):
    for c in range(c0, c1 + 1):
        yield (r0, c)
        yield (r1, c)
    for r in range(r0 + 1, r1):
        yield (r, c0)
        yield (r, c1)


def _ordered_inner_cells(inner_h: int, inner_w: int, direction: tuple[int, int]) -> list[tuple[int, int]]:
    """Local interior coordinates in head-facing order."""
    cells = [(r, c) for r in range(inner_h) for c in range(inner_w)]
    if direction == (0, 1):
        cells.sort(key=lambda p: (-p[1], p[0]))
    elif direction == (0, -1):
        cells.sort(key=lambda p: (p[1], -p[0]))
    elif direction == (1, 0):
        cells.sort(key=lambda p: (-p[0], -p[1]))
    else:  # (-1, 0)
        cells.sort(key=lambda p: (p[0], p[1]))
    return cells


def _layer_span(inner_h: int, inner_w: int, direction: tuple[int, int]) -> int:
    # Vertical motion consumes rows of width inner_w; horizontal motion consumes columns of height inner_h.
    return int(inner_w if direction[0] != 0 else inner_h)


def _grey_ordered_coords(r0: int, c0: int, inner: np.ndarray, direction: tuple[int, int]) -> list[tuple[int, int]]:
    order = _ordered_inner_cells(inner.shape[0], inner.shape[1], direction)
    return [
        (int(r0 + 1 + rr), int(c0 + 1 + cc))
        for rr, cc in order
        if int(inner[rr, cc]) == GREY
    ]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move bordered containers toward same-colored destination markers.

    Variant rule:
    - Direction still comes from head -> destination.
    - Movement steps ignore distance and depend on complete grey layers only.
      Vertical: steps = grey_count // inner_width.
      Horizontal: steps = grey_count // inner_height.
    - Recoloring consumes full layers as well, so `steps * layer_span` grey cells become orange
      from the head side inward.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    # True background is dominant non-border color (border can dominate in some cases).
    cnt = Counter(map(int, g.flatten()))
    bg = int(max((c for c in cnt.keys() if int(c) != int(BORDER)), key=lambda c: int(cnt[int(c)])))

    # --- detect containers as rectangles enclosing GREY/ORANGE with BORDER frame ---
    containers = []
    for pts in components_mask((g == GREY) | (g == ORANGE), connectivity=4):
        if pts.size == 0:
            continue
        ir0, ic0, ir1, ic1 = bbox_coords(pts)
        r0, c0, r1, c1 = int(ir0 - 1), int(ic0 - 1), int(ir1 + 1), int(ic1 + 1)
        if r0 < 0 or c0 < 0 or r1 >= H or c1 >= W:
            continue
        if any(int(g[r, c]) != BORDER for r, c in _rect_border_coords(r0, c0, r1, c1)):
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

        # Split head vs destination even if same-color pixels touch.
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

        # Add split-from-same-component destinations when head/dest touch.
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

        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        grey_coords = _grey_ordered_coords(r0, c0, inner, direction)
        layer_span = _layer_span(inner.shape[0], inner.shape[1], direction)
        steps = int(len(grey_coords) // layer_span)

        dr, dc = direction
        tr = int(dr * steps)
        tc = int(dc * steps)

        to_fill = set(grey_coords[: steps * layer_span])

        out[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
        for r, c in head_pts:
            if int(out[int(r), int(c)]) == int(hc):
                out[int(r), int(c)] = int(bg)

        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        for rr, cc in to_fill:
            pr = int(rr - r0)
            pc = int(cc - c0)
            if 0 <= pr < patch.shape[0] and 0 <= pc < patch.shape[1] and int(patch[pr, pc]) == GREY:
                patch[pr, pc] = int(ORANGE)
        paste(out, patch, int(r0 + tr), int(c0 + tc), clip=True)

        for r, c in head_pts:
            rrr = int(r + tr)
            ccc = int(c + tc)
            if 0 <= rrr < H and 0 <= ccc < W:
                out[rrr, ccc] = int(hc)

    return out


def _interior_fill_order(inner_h: int, inner_w: int, head_side: int) -> list[tuple[int, int]]:
    return _ordered_inner_cells(inner_h, inner_w, SIDE_TO_DIR[head_side])


def _side_layer_span(inner_h: int, inner_w: int, head_side: int) -> int:
    return _layer_span(inner_h, inner_w, SIDE_TO_DIR[head_side])


def _choose_mover_spec(force_remainder: bool) -> dict | None:
    head_side = random.randint(0, 3)
    inner_h = random.randint(2, 8)
    inner_w = random.randint(2, 5)
    outer_h = inner_h + 2
    outer_w = inner_w + 2

    layer_span = _side_layer_span(inner_h, inner_w, head_side)
    inner_area = inner_h * inner_w

    grey_choices = [g for g in range(layer_span, inner_area + 1) if g // layer_span >= 1]
    if force_remainder:
        grey_choices = [g for g in grey_choices if g % layer_span != 0]
    if not grey_choices:
        return None
    grey = random.choice(grey_choices)
    steps = grey // layer_span
    initial_orange = inner_area - grey

    # Keep destination beyond the moved head so motion is still visibly "toward" the marker.
    h2d = steps + random.randint(1, 6)
    gap = h2d - 1

    return {
        "inner_h": inner_h,
        "inner_w": inner_w,
        "outer_h": outer_h,
        "outer_w": outer_w,
        "head_side": head_side,
        "gap": gap,
        "head_dest_distance": h2d,
        "steps": steps,
        "initial_orange": initial_orange,
        "grey": grey,
        "layer_span": layer_span,
        "remainder": grey % layer_span,
        "is_decoy": False,
    }


def _choose_decoy_spec() -> dict:
    head_side = random.randint(0, 3)
    inner_h = random.randint(1, 8)
    inner_w = random.randint(1, 5)
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    return {
        "inner_h": inner_h,
        "inner_w": inner_w,
        "outer_h": outer_h,
        "outer_w": outer_w,
        "head_side": head_side,
        "gap": random.randint(0, 6),
        "head_dest_distance": None,
        "steps": 0,
        "initial_orange": inner_h * inner_w,
        "grey": 0,
        "layer_span": _side_layer_span(inner_h, inner_w, head_side),
        "remainder": 0,
        "is_decoy": True,
    }


def _assembly_spec(s: dict) -> dict:
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

    return {**s, "total_h": total_h, "total_w": total_w, "cont_r": cont_r, "cont_c": cont_c}


def _build_expected(grid: np.ndarray, assemblies: list[dict], placements: list[tuple[int, int]]) -> np.ndarray:
    expected = grid.copy()

    for a, (pr, pc) in zip(assemblies, placements):
        oh, ow = a["outer_h"], a["outer_w"]
        hs = a["head_side"]
        steps = a["steps"]
        inner_h, inner_w = a["inner_h"], a["inner_w"]
        initial_orange = a["initial_orange"]
        dr, dc = SIDE_TO_DIR[hs]
        layer_span = a["layer_span"]

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

        fill_order = _interior_fill_order(inner_h, inner_w, hs)
        interior = np.full((inner_h, inner_w), GREY, dtype=int)
        filled = initial_orange + steps * layer_span
        for idx_cell in range(filled):
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
    global LAST_GENERATED_META

    n_containers = random.randint(2, 4)
    n_decoys = random.choice([0, 0, 1])
    n_movers = max(1, n_containers - n_decoys)
    n_decoys = n_containers - n_movers

    specs = []
    forced_remainder_used = False
    for i in range(n_containers):
        is_decoy = i >= n_movers
        if is_decoy:
            specs.append(_choose_decoy_spec())
            continue

        force_remainder = not forced_remainder_used
        spec = _choose_mover_spec(force_remainder=force_remainder)
        if spec is None:
            return None
        forced_remainder_used = forced_remainder_used or (spec["remainder"] > 0)
        specs.append(spec)

    if not forced_remainder_used:
        return None

    assemblies = [_assembly_spec(s) for s in specs]

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

    grid = np.full((grid_h, grid_w), BG, dtype=int)

    for a, (pr, pc) in zip(assemblies, placements):
        oh, ow = a["outer_h"], a["outer_w"]
        hs = a["head_side"]
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

    expected = _build_expected(grid, assemblies, placements)
    if not np.array_equal(output, expected):
        return None

    mover_specs = [a for a in assemblies if not a["is_decoy"]]
    LAST_GENERATED_META = {
        "containers": len(assemblies),
        "movers": len(mover_specs),
        "decoys": len(assemblies) - len(mover_specs),
        "remainder_movers": sum(1 for a in mover_specs if a["remainder"] > 0),
        "old_diff_movers": sum(
            1
            for a in mover_specs
            if min(a["head_dest_distance"], a["grey"]) != a["steps"]
        ),
        "layer_spans": [a["layer_span"] for a in mover_specs],
        "steps": [a["steps"] for a in mover_specs],
        "remainders": [a["remainder"] for a in mover_specs],
        "old_steps": [min(a["head_dest_distance"], a["grey"]) for a in mover_specs],
    }
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid after 400 attempts")
