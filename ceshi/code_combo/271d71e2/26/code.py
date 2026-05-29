# Combo variant of 271d71e2 (index 26)
# combo_spec: D1/0+D8/1+D14/0
# Name: 对角移动对接留轨迹

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import random
import sys

import numpy as np
from scipy.ndimage import label

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import STRUCT8, bbox_coords, colors_present, components_mask, paste

# Design analysis for combo D1/0 + D8/1 + D14/0
# ===============================================
# a) Concrete behavior and trace
# - Keep the original container vocabulary: magenta background, black rectangular frame,
#   grey/orange interior, one same-colored head marker and one same-colored destination marker.
# - D1 changes the geometry: the head is no longer a full edge line. It is a fixed 2-pixel
#   diagonal corner marker, one of four templates:
#     NW {(r0-1,c0), (r0,c0-1)}
#     NE {(r0-1,c1), (r0,c1+1)}
#     SW {(r1,c0-1), (r1+1,c0)}
#     SE {(r1,c1+1), (r1+1,c1)}
#   The destination is an identical 2-pixel diagonal segment farther out on the same 45-degree ray.
# - Motion remains capped by grey count, but the distance is now diagonal step count:
#     steps = min(diagonal_head_to_dest_steps, grey_count)
# - Grey charging must follow the head corner, not a whole side. We therefore sort interior cells
#   by corner-distance layers and corner-specific tie breaks.
# - D8 changes the terminal branch: when `steps == diagonal_head_to_dest_steps`, the moved head lands
#   exactly on the destination and those two marker pixels turn black, visually docking into the box.
# - D14 adds a visible swept trail. For diagonal motion, reusing the orthogonal "head pixels only"
#   idea is not enough because those pixels are often swallowed by the final box. The clean readable
#   counterpart is: color every intermediate container position (excluding the final one) in the head
#   color, then paste the moved container on top. What remains visible is a stepped diagonal corridor
#   behind the box, which humans can read as a movement trail.
# - Small example:
#   a 4x4 outer box with NW head and destination 3 diagonal steps away, 5 grey cells.
#   Original task: move NW by 3, recolor 3 grey cells, head stays maroon on destination.
#   Combo task: move NW by 3, recolor the same 3 grey cells, paint the vacated diagonal swept corridor
#   maroon, and because the destination is fully reached, turn the final 2-pixel head/destination pair
#   black so it merges with the box corner.
#
# b) Complexity calibration
# - Only four diagonal directions exist, matching the original four-way branching.
# - Head and destination always reuse the same tiny repeated 2-pixel templates, so pairing stays legible.
# - Docking has one visible mode only: full reach turns the final marker pair black.
# - Trail color is always the head color and trail geometry is always "all intermediate box positions",
#   so there is one repeated visual rule instead of many width/color cases.
# - The generator forces at least one full dock and at least one trail of length >= 2 in every sample,
#   and usually also includes a partial mover or a stationary decoy, so the examples expose the cap rule,
#   the docking branch, and the trail branch together.
#
# c) Hidden assumptions broken by the combo and fixes
# - Original marker components used 4-connectivity. Diagonal head+destination gap-0 cases only connect
#   under 8-connectivity, so both the component catalog and same-component split must use STRUCT8.
# - Original head extraction assumed the closest non-bg pixels formed a full side line. That fails here.
#   Fix: detect direction by testing which of the four fixed diagonal corner templates touches the box.
# - Original destination filtering assumed one axis stayed fixed (`tc0 == 0` or `tr0 == 0`). For the
#   combo we instead require `abs(tr0) == abs(tc0)` with matching signs.
# - Original recolor order was edge-based. That would recolor the wrong cells under diagonal heads.
#   Fix: use corner-based layer ordering.
# - Original writeback assumed vacated cells become background and the final head always stays maroon.
#   Both are false now. Fix: redraw a diagonal swept trail before pasting the moved box, and overwrite
#   the final head with black when the destination is fully reached.
#
# d) Placement legality
# - The generator computes one local scenario footprint per object from the union of:
#   input box, input head, destination, final box, final head, and all intermediate box positions.
# - It then packs those axis-aligned footprints with a 1-cell background moat, ensuring in-bounds
#   placement and preventing overlaps or accidental touching between unrelated scenarios.
# - Because all trail cells lie inside the union of intermediate box positions, the reserved footprint
#   is sufficient for both input and output.
#
# e) Visual observability
# - Humans can see three coupled cues at once:
#   row and column both change, the box leaves a maroon diagonal corridor, and a full contact turns the
#   final marker pair black instead of maroon.
# - The generator guarantees these cues are present, preventing degeneration back to the base task.
#
# f) Visual cleanliness
# - The trail uses an already meaningful color (the marker color) and occupies only the box's swept area.
# - The moved box is pasted after the trail, so borders/interior remain crisp and the trail does not hide
#   the final object.
# - Objects are separated by a moat, keeping same-colored markers from unrelated scenarios visually apart.
#
# g) Color-change inferability
# - There are still only the original structural colors plus one conditional output recolor:
#   grey->orange by charge order, and maroon->black only on full docking.
# - Both recolors are spatially anchored: orange grows from the head corner inward, and black appears
#   exactly where the head meets the destination.
#
# New solver assumptions and generator support
# -------------------------------------------
# - Every head is exactly one of the four diagonal 2-pixel templates above.
# - Every destination is an identical template on the same diagonal ray.
# - Nearest valid destination selection is unambiguous because the generator validates each candidate
#   against the solver and resamples ambiguous placements away.
# - The trail rule assumes intermediate corridor cells are free of unrelated objects; the packed local
#   footprints enforce that.

BG = 6
BORDER = 0
GREY = 5
ORANGE = 7
HEAD_COLOR = 9

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def _normalize_points(pts: set[tuple[int, int]]) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    coords = np.array(list(pts), dtype=int)
    r0, c0, _r1, _c1 = bbox_coords(coords)
    shape = {(int(r - r0), int(c - c0)) for r, c in pts}
    return shape, (int(r0), int(c0))


def _shift_points(pts: set[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(int(r + dr), int(c + dc)) for r, c in pts}


def _rect_border_coords(r0: int, c0: int, r1: int, c1: int):
    for c in range(c0, c1 + 1):
        yield (r0, c)
        yield (r1, c)
    for r in range(r0 + 1, r1):
        yield (r, c0)
        yield (r, c1)


def _head_pts_for_box(box: tuple[int, int, int, int], direction: tuple[int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    if direction == (-1, -1):
        return {(r0 - 1, c0), (r0, c0 - 1)}
    if direction == (-1, 1):
        return {(r0 - 1, c1), (r0, c1 + 1)}
    if direction == (1, -1):
        return {(r1, c0 - 1), (r1 + 1, c0)}
    return {(r1, c1 + 1), (r1 + 1, c1)}


def _head_local_points(outer_h: int, outer_w: int, direction: tuple[int, int]) -> set[tuple[int, int]]:
    if direction == (-1, -1):
        return {(-1, 0), (0, -1)}
    if direction == (-1, 1):
        return {(-1, outer_w - 1), (0, outer_w)}
    if direction == (1, -1):
        return {(outer_h - 1, -1), (outer_h, 0)}
    return {(outer_h - 1, outer_w), (outer_h, outer_w - 1)}


def _corner_fill_sort_key(
    inner_h: int, inner_w: int, direction: tuple[int, int], cell: tuple[int, int]
) -> tuple[int, int, int]:
    r, c = cell
    if direction == (-1, -1):
        return (int(r + c), int(r), int(c))
    if direction == (-1, 1):
        return (int(r + (inner_w - 1 - c)), int(r), int(-c))
    if direction == (1, -1):
        return (int((inner_h - 1 - r) + c), int(-r), int(c))
    return (int((inner_h - 1 - r) + (inner_w - 1 - c)), int(-r), int(-c))


def _corner_fill_order(inner_h: int, inner_w: int, direction: tuple[int, int]) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(inner_h) for c in range(inner_w)]
    cells.sort(key=lambda p: _corner_fill_sort_key(inner_h, inner_w, direction, p))
    return cells


def _box_points(outer_h: int, outer_w: int, top_left: tuple[int, int]) -> set[tuple[int, int]]:
    r0, c0 = top_left
    return {
        (int(r0 + rr), int(c0 + cc))
        for rr in range(outer_h)
        for cc in range(outer_w)
    }


def _paint_diag_trail(
    out: np.ndarray,
    box: tuple[int, int, int, int],
    direction: tuple[int, int],
    steps: int,
    color: int,
):
    if steps <= 0:
        return
    r0, c0, r1, c1 = box
    outer_h = int(r1 - r0 + 1)
    outer_w = int(c1 - c0 + 1)
    dr, dc = direction
    for k in range(steps):
        rr0 = int(r0 + dr * k)
        cc0 = int(c0 + dc * k)
        out[rr0 : rr0 + outer_h, cc0 : cc0 + outer_w] = int(color)


def _trail_visible_cell_count(spec: dict) -> int:
    direction = spec["direction"]
    steps = spec["steps"]
    if steps <= 0:
        return 0
    old_union: set[tuple[int, int]] = set()
    for k in range(steps):
        old_union |= _box_points(
            spec["outer_h"],
            spec["outer_w"],
            (direction[0] * k, direction[1] * k),
        )
    final_box = _box_points(
        spec["outer_h"],
        spec["outer_w"],
        (direction[0] * steps, direction[1] * steps),
    )
    return int(len(old_union - final_box))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move bordered containers toward same-colored diagonal destination markers.

    Combo behavior:
    - diagonal corner markers determine one of four 45-degree movement rays
    - grey cells turn orange from the head corner inward
    - every intermediate box position leaves a head-colored trail
    - on full contact, the moved head merges with the destination as black border
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
        labc, nc = label(g == int(col), structure=STRUCT8)
        for k in range(1, int(nc) + 1):
            coords = np.argwhere(labc == int(k))
            if coords.size == 0:
                continue
            pts = {(int(r), int(c)) for r, c in coords}
            shape, tl = _normalize_points(pts)
            comp_by_color[int(col)].append({"pts": pts, "shape": shape, "tl": tl})

    def pick_head_for_container(box_: tuple[int, int, int, int]):
        r0, c0, r1, c1 = box_
        head_seeds = []
        for r, c in _rect_border_coords(r0, c0, r1, c1):
            for dr, dc in DIR4:
                rr, cc = int(r + dr), int(c + dc)
                if 0 <= rr < H and 0 <= cc < W:
                    val = int(g[rr, cc])
                    if val not in (int(bg), int(BORDER), int(GREY), int(ORANGE)):
                        head_seeds.append((rr, cc, val))
        if not head_seeds:
            return None

        hc = int(Counter(val for _r, _c, val in head_seeds).most_common(1)[0][0])
        seed_pts = {(int(r), int(c)) for r, c, val in head_seeds if int(val) == int(hc)}

        comp0 = None
        for comp in comp_by_color.get(hc, []):
            if comp["pts"] & seed_pts:
                comp0 = comp
                break
        if comp0 is None:
            return None

        directions = [d for d in DIAG_DIRS if _head_pts_for_box(box_, d) <= seed_pts]
        if len(directions) != 1:
            return None
        direction = directions[0]
        head_pts = _head_pts_for_box(box_, direction)
        if not head_pts <= comp0["pts"]:
            return None

        head_shape, head_tl = _normalize_points(head_pts)
        return {
            "color": hc,
            "head_pts": head_pts,
            "head_shape": head_shape,
            "head_tl": head_tl,
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

        dests = []
        for comp in comp_by_color.get(hc, []):
            if comp["shape"] != head_shape or comp["pts"] == comp_pts:
                continue
            dests.append(comp)

        rem = comp_pts - head_pts
        if rem:
            mask_rem = np.zeros((H, W), dtype=bool)
            for rr, cc in rem:
                mask_rem[int(rr), int(cc)] = True
            labr, nr = label(mask_rem, structure=STRUCT8)
            for k in range(1, int(nr) + 1):
                pts = {(int(r), int(c)) for r, c in np.argwhere(labr == int(k))}
                if not pts:
                    continue
                shape, tl = _normalize_points(pts)
                if shape != head_shape:
                    continue
                dests.append({"pts": pts, "shape": shape, "tl": tl})

        if not dests:
            continue

        best_dest = None
        best_dist = None
        for dest in dests:
            tr0 = int(dest["tl"][0] - head_tl[0])
            tc0 = int(dest["tl"][1] - head_tl[1])
            if tr0 == 0 or abs(tr0) != abs(tc0):
                continue
            if int(tr0 / abs(tr0)) != int(direction[0]):
                continue
            if int(tc0 / abs(tc0)) != int(direction[1]):
                continue
            dist = abs(tr0)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_dest = dest
        if best_dest is None or best_dist is None:
            continue

        dock_distance = int(best_dist)
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        grey_local = [
            (int(rr), int(cc))
            for rr in range(inner.shape[0])
            for cc in range(inner.shape[1])
            if int(inner[rr, cc]) == int(GREY)
        ]
        steps = min(int(dock_distance), int(len(grey_local)))
        full_dock = steps == int(dock_distance) and steps > 0

        dr, dc = direction
        tr = int(dr * steps)
        tc = int(dc * steps)

        grey_local.sort(key=lambda p: _corner_fill_sort_key(inner.shape[0], inner.shape[1], direction, p))
        to_fill_local = set(grey_local[:steps])

        out[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
        for r, c in head_pts:
            if 0 <= r < H and 0 <= c < W and int(out[int(r), int(c)]) == int(hc):
                out[int(r), int(c)] = int(bg)

        _paint_diag_trail(out, (r0, c0, r1, c1), direction, steps, hc)

        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        for rr, cc in to_fill_local:
            patch[int(rr + 1), int(cc + 1)] = int(ORANGE)
        paste(out, patch, int(r0 + tr), int(c0 + tc), clip=True)

        final_head_color = BORDER if full_dock else hc
        for r, c in head_pts:
            rrr = int(r + tr)
            ccc = int(c + tc)
            if 0 <= rrr < H and 0 <= ccc < W:
                out[int(rrr), int(ccc)] = int(final_head_color)

    return out


def _scenario_bbox(spec: dict) -> dict:
    oh, ow = spec["outer_h"], spec["outer_w"]
    direction = spec["direction"]
    h2d = spec["h2d"]
    steps = spec["steps"]

    container_pts = _box_points(oh, ow, (0, 0))
    head_pts = _head_local_points(oh, ow, direction)
    dest_pts = _shift_points(head_pts, direction[0] * h2d, direction[1] * h2d)
    out_container_pts = _shift_points(container_pts, direction[0] * steps, direction[1] * steps)
    out_head_pts = _shift_points(head_pts, direction[0] * steps, direction[1] * steps)

    trail_pts: set[tuple[int, int]] = set()
    for k in range(steps):
        trail_pts |= _box_points(oh, ow, (direction[0] * k, direction[1] * k))

    footprint = container_pts | head_pts | dest_pts | out_container_pts | out_head_pts | trail_pts
    rows = [r for r, _c in footprint]
    cols = [c for _r, c in footprint]

    spec = dict(spec)
    spec["min_r"] = int(min(rows))
    spec["min_c"] = int(min(cols))
    spec["asm_h"] = int(max(rows) - min(rows) + 1)
    spec["asm_w"] = int(max(cols) - min(cols) + 1)
    spec["visible_trail_cells"] = _trail_visible_cell_count(spec)
    return spec


def _choose_active_modes(n_active: int) -> list[str]:
    modes = ["dock_long"]
    if n_active >= 2:
        modes.append("partial")
    while len(modes) < n_active:
        modes.append(random.choice(["dock", "partial"]))
    random.shuffle(modes)
    return modes


def _sample_dimensions(min_area: int) -> tuple[int, int]:
    while True:
        inner_h = random.randint(1, 7)
        inner_w = random.randint(1, 5)
        if inner_h * inner_w >= min_area:
            return inner_h, inner_w


def _expected_output_for_spec(expected: np.ndarray, spec: dict, placement: tuple[int, int]):
    direction = spec["direction"]
    steps = spec["steps"]
    if steps == 0:
        return

    oh, ow = spec["outer_h"], spec["outer_w"]
    inner_h, inner_w = spec["inner_h"], spec["inner_w"]
    initial_orange = spec["initial_orange"]
    shift_r = int(placement[0] - spec["min_r"])
    shift_c = int(placement[1] - spec["min_c"])
    cr, cc = int(shift_r), int(shift_c)
    head_pts = _shift_points(_head_local_points(oh, ow, direction), shift_r, shift_c)

    expected[cr : cr + oh, cc : cc + ow] = int(BG)
    for r, c in head_pts:
        expected[int(r), int(c)] = int(BG)

    _paint_diag_trail(expected, (cr, cc, cr + oh - 1, cc + ow - 1), direction, steps, HEAD_COLOR)

    nr = int(cr + direction[0] * steps)
    nc = int(cc + direction[1] * steps)
    patch = np.full((oh, ow), BORDER, dtype=int)
    interior = np.full((inner_h, inner_w), GREY, dtype=int)
    for rr, cc2 in _corner_fill_order(inner_h, inner_w, direction)[: initial_orange + steps]:
        interior[int(rr), int(cc2)] = int(ORANGE)
    patch[1:-1, 1:-1] = interior
    expected[nr : nr + oh, nc : nc + ow] = patch

    final_head_color = BORDER if steps == spec["h2d"] else HEAD_COLOR
    moved_head = _shift_points(head_pts, direction[0] * steps, direction[1] * steps)
    for r, c in moved_head:
        expected[int(r), int(c)] = int(final_head_color)


def _gen_attempt(return_meta: bool = False):
    n_containers = random.randint(2, 4)
    n_decoys = random.choice([0, 0, 1])
    n_active = max(1, n_containers - n_decoys)
    n_decoys = n_containers - n_active

    active_modes = _choose_active_modes(n_active)
    specs = []

    for idx in range(n_containers):
        is_decoy = idx >= n_active
        mode = "decoy" if is_decoy else active_modes[idx]
        min_area = 2 if mode == "dock_long" else 1
        inner_h, inner_w = _sample_dimensions(min_area)
        inner_area = inner_h * inner_w
        direction = random.choice(DIAG_DIRS)
        outer_h = inner_h + 2
        outer_w = inner_w + 2

        if mode == "decoy":
            grey = 0
            initial_orange = inner_area
            h2d = random.randint(1, 6)
            steps = 0
        elif mode == "dock_long":
            h2d = random.randint(2, min(8, inner_area))
            grey = random.randint(h2d, inner_area)
            initial_orange = inner_area - grey
            steps = h2d
        elif mode == "dock":
            h2d = random.randint(1, min(8, inner_area))
            grey = random.randint(h2d, inner_area)
            initial_orange = inner_area - grey
            steps = h2d
        else:  # partial
            h2d = random.randint(2, 8)
            grey = random.randint(1, min(inner_area, h2d - 1))
            initial_orange = inner_area - grey
            steps = grey

        specs.append(
            {
                "inner_h": inner_h,
                "inner_w": inner_w,
                "outer_h": outer_h,
                "outer_w": outer_w,
                "direction": direction,
                "h2d": h2d,
                "steps": steps,
                "grey": grey,
                "initial_orange": initial_orange,
                "mode": mode,
                "is_decoy": is_decoy,
            }
        )

    specs = [_scenario_bbox(spec) for spec in specs]

    if not any(spec["steps"] == spec["h2d"] and spec["steps"] > 0 for spec in specs):
        return None
    if max(int(spec["visible_trail_cells"]) for spec in specs) < 2:
        return None

    total_area = sum(spec["asm_h"] * spec["asm_w"] for spec in specs)
    min_side = max(10, int(total_area ** 0.5) + 4)
    max_asm_h = max(spec["asm_h"] for spec in specs)
    max_asm_w = max(spec["asm_w"] for spec in specs)
    lo_h = max(10, max_asm_h + 2)
    lo_w = max(10, max_asm_w + 2)
    hi_h = min(26, max(lo_h, min_side + 6))
    hi_w = min(26, max(lo_w, min_side + 6))
    if lo_h > 26 or lo_w > 26:
        return None

    grid_h = random.randint(lo_h, hi_h)
    grid_w = random.randint(lo_w, hi_w)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * n_containers

    order = sorted(range(n_containers), key=lambda i: specs[i]["asm_h"] * specs[i]["asm_w"], reverse=True)
    for idx in order:
        spec = specs[idx]
        ah, aw = spec["asm_h"], spec["asm_w"]
        if ah > grid_h or aw > grid_w:
            return None
        placed = False
        for _ in range(400):
            pr = random.randint(0, grid_h - ah)
            pc = random.randint(0, grid_w - aw)
            rr0 = max(0, pr - 1)
            cc0 = max(0, pc - 1)
            rr1 = min(grid_h, pr + ah + 1)
            cc1 = min(grid_w, pc + aw + 1)
            if not occupied[rr0:rr1, cc0:cc1].any():
                occupied[pr : pr + ah, pc : pc + aw] = True
                placements[idx] = (pr, pc)
                placed = True
                break
        if not placed:
            return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)

    for spec, placement in zip(specs, placements):
        assert placement is not None
        pr, pc = placement
        direction = spec["direction"]
        oh, ow = spec["outer_h"], spec["outer_w"]
        inner_h, inner_w = spec["inner_h"], spec["inner_w"]
        shift_r = int(pr - spec["min_r"])
        shift_c = int(pc - spec["min_c"])
        cr, cc = int(shift_r), int(shift_c)

        grid[cr, cc : cc + ow] = int(BORDER)
        grid[cr + oh - 1, cc : cc + ow] = int(BORDER)
        grid[cr : cr + oh, cc] = int(BORDER)
        grid[cr : cr + oh, cc + ow - 1] = int(BORDER)

        interior = np.full((inner_h, inner_w), GREY, dtype=int)
        for rr, cc2 in _corner_fill_order(inner_h, inner_w, direction)[: spec["initial_orange"]]:
            interior[int(rr), int(cc2)] = int(ORANGE)
        grid[cr + 1 : cr + 1 + inner_h, cc + 1 : cc + 1 + inner_w] = interior

        head_pts = _shift_points(_head_local_points(oh, ow, direction), shift_r, shift_c)
        dest_pts = _shift_points(head_pts, direction[0] * spec["h2d"], direction[1] * spec["h2d"])
        for r, c in head_pts | dest_pts:
            if not (0 <= r < grid_h and 0 <= c < grid_w):
                return None
            grid[int(r), int(c)] = int(HEAD_COLOR)

    try:
        output = main(grid)
    except Exception:
        return None

    expected = grid.copy()
    for spec, placement in zip(specs, placements):
        assert placement is not None
        _expected_output_for_spec(expected, spec, placement)

    if not np.array_equal(output, expected):
        return None

    if return_meta:
        return grid, specs, placements
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid combo sample after 400 attempts")
