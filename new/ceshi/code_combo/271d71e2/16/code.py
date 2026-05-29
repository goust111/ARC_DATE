# Combo variant of 271d71e2 (index 16)
# combo_spec: D11/2+D15/2
# Name: L形目标加边框厚度编码

from __future__ import annotations

# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior trace
# - Original task:
#   A 1-pixel black rectangular container had a maroon head line on one side and
#   a same-shape maroon destination line farther away on the movement axis.
#   Travel distance came from the visible head->destination offset.
# - This combo:
#   1. The destination is no longer a line. It is a maroon L or T marker.
#      The marker determines which head cell is the alignment anchor:
#        * T: use the head midpoint and the T-junction.
#        * L: use the head endpoint corresponding to the arm side and the elbow.
#   2. Travel distance no longer comes from visible marker spacing. Every marker
#      anchor is placed at the same visible offset MARKER_DISTANCE, while the
#      black border thickness t encodes the old gap, so steps = min(t, grey_count).
# - Small example:
#   Two top-headed containers may both have their marker anchor 4 cells above the
#   relevant head anchor. One has thickness 1, one has thickness 4, and both have
#   enough grey cells. The thin one moves up by 1, the thick one by 4. So shape
#   is needed to know which head cell the marker refers to, and thickness is
#   needed to know how far the container actually travels.
#
# b) Complexity calibration
# - Only two marker families are used: L and T.
# - Thickness is limited to 1..4, so the encoded distance is visible but not too
#   high-dimensional.
# - The visible marker-anchor offset is fixed globally, which removes the old
#   direct distance cue and forces the solver to use thickness.
# - Each generated grid uses at least two movers, mixed L/T markers, and distinct
#   thicknesses, so the fusion signal repeats inside one scene.
#
# c) Original implicit assumptions that break
# - `inner_bbox +- 1` no longer finds the container: thick borders require
#   recovering the maximal black frame thickness around each full interior block.
# - `destination.shape == head.shape` no longer works: markers are L/T shapes.
# - `steps = visible head->dest offset` is invalid: movement is thickness-coded.
# - A simple nearest maroon line match is unsafe: we must parse marker anchors and
#   require the fixed visible offset to avoid cross-object accidental matches.
#
# d) Placement legality
# - Each object is built as an input/output union patch whose bbox contains:
#     input container + input head + fixed-position marker + output container + output head
#   The generator places these patches with a 1-cell background moat, so objects
#   stay non-overlapping in both input and output.
# - After placement, we explicitly reject layouts where any container's head anchor
#   could also match another marker at the required direction and fixed offset.
#
# e) Visual observability
# - Human-solvable cues:
#   * L vs T visibly changes which part of the head is "pointed at".
#   * Visible marker spacing repeats, while movement amount varies with border thickness.
# - At least one mover per sample is thickness-controlled in a way that would
#   disagree with the original distance rule (`grey > thickness` and `thickness < 4`).
#
# f) Visual cleanliness
# - No extra colors are introduced.
# - Markers remain small connected symbols separated from containers and from one
#   another by background padding.
# - The maroon head and marker never touch in the input, so the two roles stay legible.
#
# g) Change inferability
# - Color changes remain the original structured grey->orange charging order.
# - The fusion only adds structural cues (marker shape and border thickness), both
#   directly visible in the pixels.
#
# h) New solver assumptions and generator guarantees
# - Every container has a full-edge 1-pixel maroon head adjacent to exactly one side.
# - Every destination marker is a separate maroon component analyzable as an L or T.
# - The relevant marker anchor is exactly MARKER_DISTANCE cells ahead of the chosen
#   head anchor along the movement axis; the generator enforces this.
# - T markers need an odd head length so the midpoint is unique; the generator
#   enforces the necessary parity on the outer container span.
# - Movement uses only detected thickness, so the generator must include at least
#   one case where thickness meaningfully differs from the visible marker spacing.
# - Every sample also includes at least one L-marker mover whose chosen endpoint
#   is not the default first cell of the head line, so the L-anchor rule is
#   behaviorally active rather than merely decorative.

from collections import Counter, defaultdict
from pathlib import Path
import random
import sys

import numpy as np
from scipy.ndimage import label

SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import STRUCT4, bbox_coords, colors_present, components_mask, paste


BG = 6
BORDER = 0
GREY = 5
ORANGE = 7
HEAD_COLOR = 9

MIN_THICKNESS = 1
MAX_THICKNESS = 4
MARKER_DISTANCE = MAX_THICKNESS

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDE_TO_DIR = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
PERP_DIRS = {
    (-1, 0): ((0, -1), (0, 1)),
    (1, 0): ((0, -1), (0, 1)),
    (0, -1): ((-1, 0), (1, 0)),
    (0, 1): ((-1, 0), (1, 0)),
}


def rect_border_coords(r0: int, c0: int, r1: int, c1: int):
    for c in range(c0, c1 + 1):
        yield (r0, c)
        yield (r1, c)
    for r in range(r0 + 1, r1):
        yield (r, c0)
        yield (r, c1)


def _ray(anchor: tuple[int, int], delta: tuple[int, int], length: int) -> list[tuple[int, int]]:
    ar, ac = anchor
    dr, dc = delta
    return [(int(ar + dr * k), int(ac + dc * k)) for k in range(length)]


def _ray_len(pts: set[tuple[int, int]], anchor: tuple[int, int], delta: tuple[int, int]) -> int:
    r, c = anchor
    dr, dc = delta
    out = 0
    while (int(r), int(c)) in pts:
        out += 1
        r += dr
        c += dc
    return int(out)


def _marker_cells_from_spec(
    anchor: tuple[int, int],
    direction: tuple[int, int],
    marker_kind: str,
    stem_len: int,
    arm_dir: tuple[int, int] | None = None,
    arm_len: int | None = None,
) -> set[tuple[int, int]]:
    cells = set(_ray(anchor, direction, stem_len))
    if marker_kind == "T":
        perp_a, perp_b = PERP_DIRS[direction]
        cells.add((int(anchor[0] + perp_a[0]), int(anchor[1] + perp_a[1])))
        cells.add((int(anchor[0] + perp_b[0]), int(anchor[1] + perp_b[1])))
    else:
        assert arm_dir is not None and arm_len is not None
        cells.update(_ray(anchor, arm_dir, arm_len))
    return cells


def _analyze_marker_component(pts: set[tuple[int, int]]) -> dict[tuple[int, int], dict]:
    analyses: dict[tuple[int, int], dict] = {}
    for direction in DIR4:
        dr, dc = direction
        perp_a, perp_b = PERP_DIRS[direction]
        candidates = []
        for anchor in pts:
            ar, ac = anchor
            if (int(ar + dr), int(ac + dc)) not in pts:
                continue
            if (int(ar - dr), int(ac - dc)) in pts:
                continue
            stem_len = _ray_len(pts, anchor, direction)

            pa = (int(ar + perp_a[0]), int(ac + perp_a[1]))
            pb = (int(ar + perp_b[0]), int(ac + perp_b[1]))
            if pa in pts and pb in pts:
                cells = set(_ray(anchor, direction, stem_len))
                cells.add(pa)
                cells.add(pb)
                if cells == pts:
                    candidates.append({"kind": "T", "anchor": anchor})

            for perp in (perp_a, perp_b):
                pr, pc = int(ar + perp[0]), int(ac + perp[1])
                rr, cc = int(ar - perp[0]), int(ac - perp[1])
                if (pr, pc) not in pts or (rr, cc) in pts:
                    continue
                arm_len = _ray_len(pts, anchor, perp)
                cells = set(_ray(anchor, direction, stem_len))
                cells.update(_ray(anchor, perp, arm_len))
                if cells == pts:
                    candidates.append({"kind": "L", "anchor": anchor, "perp": perp})

        dedup = []
        seen = set()
        for cand in candidates:
            key = (cand["kind"], cand["anchor"], cand.get("perp"))
            if key not in seen:
                seen.add(key)
                dedup.append(cand)
        if len(dedup) == 1:
            analyses[direction] = dedup[0]
    return analyses


def _head_anchor_for_marker(
    head_pts: set[tuple[int, int]],
    direction: tuple[int, int],
    marker: dict,
) -> tuple[int, int] | None:
    if direction[0] != 0:
        cells = sorted(head_pts, key=lambda p: (p[1], p[0]))
    else:
        cells = sorted(head_pts, key=lambda p: (p[0], p[1]))

    if marker["kind"] == "T":
        if len(cells) % 2 == 0:
            return None
        return cells[len(cells) // 2]

    perp = marker["perp"]
    if direction[0] != 0:
        return cells[0] if perp == (0, 1) else cells[-1]
    return cells[0] if perp == (1, 0) else cells[-1]


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


def _max_black_frame_thickness(g: np.ndarray, ir0: int, ic0: int, ir1: int, ic1: int) -> int:
    H, W = g.shape
    thickness = 0
    while True:
        t = thickness + 1
        r0, c0 = ir0 - t, ic0 - t
        r1, c1 = ir1 + t, ic1 + t
        if r0 < 0 or c0 < 0 or r1 >= H or c1 >= W:
            break
        if all(int(g[r, c]) == BORDER for r, c in rect_border_coords(r0, c0, r1, c1)):
            thickness = t
        else:
            break
    return int(thickness)


def _detect_containers(g: np.ndarray) -> list[dict]:
    containers = []
    seen = set()
    for pts in components_mask((g == GREY) | (g == ORANGE), connectivity=4):
        if pts.size == 0:
            continue
        ir0, ic0, ir1, ic1 = bbox_coords(pts)
        inner = g[ir0 : ir1 + 1, ic0 : ic1 + 1]
        if pts.shape[0] != inner.size:
            continue
        if not bool(np.isin(inner, [GREY, ORANGE]).all()):
            continue

        thickness = _max_black_frame_thickness(g, int(ir0), int(ic0), int(ir1), int(ic1))
        if thickness < MIN_THICKNESS:
            continue

        r0, c0 = int(ir0 - thickness), int(ic0 - thickness)
        r1, c1 = int(ir1 + thickness), int(ic1 + thickness)
        outer = g[r0 : r1 + 1, c0 : c1 + 1]
        band_mask = np.ones(outer.shape, dtype=bool)
        band_mask[thickness : thickness + inner.shape[0], thickness : thickness + inner.shape[1]] = False
        if not bool((outer[band_mask] == BORDER).all()):
            continue

        key = (r0, c0, r1, c1)
        if key in seen:
            continue
        seen.add(key)
        containers.append(
            {
                "outer_box": key,
                "inner_box": (int(ir0), int(ic0), int(ir1), int(ic1)),
                "thickness": int(thickness),
            }
        )
    return containers


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape

    cnt = Counter(map(int, g.flatten()))
    bg = int(max((c for c in cnt.keys() if int(c) != BORDER), key=lambda c: int(cnt[int(c)])))
    out = g.copy()

    containers = _detect_containers(g)
    assert containers

    all_head_colors = [c for c in colors_present(g, exclude={bg, BORDER, GREY, ORANGE})]

    comp_by_color: dict[int, list[dict]] = defaultdict(list)
    for col in all_head_colors:
        labc, nc = label(g == int(col), structure=STRUCT4)
        for k in range(1, int(nc) + 1):
            coords = np.argwhere(labc == int(k))
            if coords.size == 0:
                continue
            pts = {(int(r), int(c)) for r, c in coords}
            r0, c0, _r1, _c1 = bbox_coords(coords)
            comp_by_color[int(col)].append(
                {
                    "id": int(k),
                    "pts": pts,
                    "markers": _analyze_marker_component(pts),
                    "tl": (int(r0), int(c0)),
                }
            )

    def pick_head_for_container(container: dict) -> dict | None:
        r0, c0, r1, c1 = container["outer_box"]
        head_seeds = []
        for r, c in rect_border_coords(r0, c0, r1, c1):
            for dr, dc in DIR4:
                rr, cc = int(r + dr), int(c + dc)
                if 0 <= rr < H and 0 <= cc < W:
                    val = int(g[rr, cc])
                    if val not in (bg, BORDER, GREY, ORANGE):
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

        return {
            "color": hc,
            "head_pts": head_pts,
            "comp_id": int(comp0["id"]),
            "dir": direction,
        }

    for container in containers:
        r0, c0, r1, c1 = container["outer_box"]
        ir0, ic0, ir1, ic1 = container["inner_box"]

        head_info = pick_head_for_container(container)
        if head_info is None:
            continue
        hc = int(head_info["color"])
        head_pts = set(head_info["head_pts"])
        direction = head_info["dir"]
        if direction == (0, 0):
            continue

        marker_match = None
        for comp in comp_by_color.get(hc, []):
            if int(comp["id"]) == int(head_info["comp_id"]):
                continue
            marker = comp["markers"].get(direction)
            if marker is None:
                continue
            head_anchor = _head_anchor_for_marker(head_pts, direction, marker)
            if head_anchor is None:
                continue

            hr, hc0 = head_anchor
            ar, ac = marker["anchor"]
            if direction[0] != 0:
                if int(ac) != int(hc0):
                    continue
                delta = int(ar - hr)
                if int(direction[0]) * delta <= 0 or abs(delta) != MARKER_DISTANCE:
                    continue
            else:
                if int(ar) != int(hr):
                    continue
                delta = int(ac - hc0)
                if int(direction[1]) * delta <= 0 or abs(delta) != MARKER_DISTANCE:
                    continue

            if marker_match is not None:
                marker_match = None
                break
            marker_match = (marker, head_anchor)

        if marker_match is None:
            continue

        inner = g[ir0 : ir1 + 1, ic0 : ic1 + 1]
        grey_coords = [
            (int(ir0 + rr), int(ic0 + cc))
            for rr in range(inner.shape[0])
            for cc in range(inner.shape[1])
            if int(inner[rr, cc]) == GREY
        ]
        steps = min(int(container["thickness"]), int(len(grey_coords)))

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
            if int(out[int(r), int(c)]) == int(hc):
                out[int(r), int(c)] = int(bg)

        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        for rr, cc in to_fill:
            pr = int(rr - r0)
            pc = int(cc - c0)
            if int(patch[pr, pc]) == GREY:
                patch[pr, pc] = int(ORANGE)
        paste(out, patch, int(r0 + tr), int(c0 + tc), clip=True)

        for r, c in head_pts:
            rr = int(r + tr)
            cc = int(c + tc)
            if 0 <= rr < H and 0 <= cc < W:
                out[rr, cc] = int(hc)

    return out


def _choose_marker_kinds(n_containers: int) -> list[str]:
    kinds = ["L", "T"]
    while len(kinds) < n_containers:
        kinds.append(random.choice(["L", "T"]))
    random.shuffle(kinds)
    return kinds[:n_containers]


def _choose_inner_dims(marker_kind: str, head_side: int, max_inner_h: int, max_inner_w: int) -> tuple[int, int]:
    if marker_kind == "T" and head_side in (0, 1):
        inner_w = random.choice([w for w in range(1, max_inner_w + 1) if w % 2 == 1])
        inner_h = random.randint(1, max_inner_h)
    elif marker_kind == "T":
        inner_h = random.choice([h for h in range(1, max_inner_h + 1) if h % 2 == 1])
        inner_w = random.randint(1, max_inner_w)
    else:
        inner_h = random.randint(1, max_inner_h)
        inner_w = random.randint(1, max_inner_w)
    return int(inner_h), int(inner_w)


def _make_spec(marker_kind: str, is_decoy: bool, max_inner_h: int, max_inner_w: int) -> dict:
    head_side = random.randint(0, 3)
    inner_h, inner_w = _choose_inner_dims(marker_kind, head_side, max_inner_h, max_inner_w)
    thickness = random.randint(MIN_THICKNESS, MAX_THICKNESS)
    inner_area = int(inner_h * inner_w)
    outer_h = int(inner_h + 2 * thickness)
    outer_w = int(inner_w + 2 * thickness)

    if is_decoy:
        grey = 0
        initial_orange = inner_area
        steps = 0
        cap_type = "decoy"
    else:
        if inner_area <= thickness:
            grey = random.randint(1, inner_area)
        else:
            if random.random() < 0.6:
                grey = random.randint(thickness + 1, inner_area)
            else:
                grey = random.randint(1, inner_area)
        initial_orange = inner_area - grey
        steps = min(grey, thickness)
        cap_type = "thickness" if grey > thickness else "grey"

    spec = {
        "marker_kind": marker_kind,
        "head_side": head_side,
        "inner_h": int(inner_h),
        "inner_w": int(inner_w),
        "inner_area": int(inner_area),
        "thickness": int(thickness),
        "outer_h": int(outer_h),
        "outer_w": int(outer_w),
        "grey": int(grey),
        "initial_orange": int(initial_orange),
        "steps": int(steps),
        "is_decoy": bool(is_decoy),
        "cap_type": cap_type,
        "stem_len": random.randint(2, 4),
    }
    if marker_kind == "L":
        spec["arm_dir"] = random.choice(list(PERP_DIRS[SIDE_TO_DIR[head_side]]))
        spec["arm_len"] = random.randint(2, 3)
    return spec


def _layout_from_spec(spec: dict) -> dict:
    oh, ow = int(spec["outer_h"]), int(spec["outer_w"])
    hs = int(spec["head_side"])
    direction = SIDE_TO_DIR[hs]

    container_cells = {(r, c) for r in range(oh) for c in range(ow)}
    if hs == 0:
        head_cells = {(-1, c) for c in range(ow)}
    elif hs == 1:
        head_cells = {(oh, c) for c in range(ow)}
    elif hs == 2:
        head_cells = {(r, -1) for r in range(oh)}
    else:
        head_cells = {(r, ow) for r in range(oh)}

    if hs in (0, 1):
        head_order = sorted(head_cells, key=lambda p: (p[1], p[0]))
    else:
        head_order = sorted(head_cells, key=lambda p: (p[0], p[1]))

    if spec["marker_kind"] == "T":
        head_anchor = head_order[len(head_order) // 2]
    else:
        arm_dir = spec["arm_dir"]
        if hs in (0, 1):
            head_anchor = head_order[0] if arm_dir == (0, 1) else head_order[-1]
        else:
            head_anchor = head_order[0] if arm_dir == (1, 0) else head_order[-1]

    marker_anchor = (
        int(head_anchor[0] + direction[0] * MARKER_DISTANCE),
        int(head_anchor[1] + direction[1] * MARKER_DISTANCE),
    )
    marker_cells = _marker_cells_from_spec(
        marker_anchor,
        direction,
        spec["marker_kind"],
        int(spec["stem_len"]),
        spec.get("arm_dir"),
        spec.get("arm_len"),
    )

    steps = int(spec["steps"])
    dr, dc = direction
    output_container = {(int(r + dr * steps), int(c + dc * steps)) for r, c in container_cells}
    output_head = {(int(r + dr * steps), int(c + dc * steps)) for r, c in head_cells}

    all_cells = container_cells | head_cells | marker_cells | output_container | output_head
    rows = [r for r, _c in all_cells]
    cols = [c for _r, c in all_cells]
    shift_r = -int(min(rows))
    shift_c = -int(min(cols))

    def sh(pt: tuple[int, int]) -> tuple[int, int]:
        return (int(pt[0] + shift_r), int(pt[1] + shift_c))

    layout = dict(spec)
    layout["total_h"] = int(max(rows) - min(rows) + 1)
    layout["total_w"] = int(max(cols) - min(cols) + 1)
    layout["cont_r"] = int(shift_r)
    layout["cont_c"] = int(shift_c)
    layout["head_cells_local"] = {sh(p) for p in head_cells}
    layout["head_anchor_local"] = sh(head_anchor)
    layout["marker_cells_local"] = {sh(p) for p in marker_cells}
    layout["marker_anchor_local"] = sh(marker_anchor)
    return layout


def _draw_container(
    grid: np.ndarray,
    top: int,
    left: int,
    inner_h: int,
    inner_w: int,
    thickness: int,
    head_side: int,
    orange_count: int,
) -> None:
    outer_h = inner_h + 2 * thickness
    outer_w = inner_w + 2 * thickness
    grid[int(top) : int(top + outer_h), int(left) : int(left + outer_w)] = int(BORDER)

    interior = np.full((inner_h, inner_w), GREY, dtype=int)
    fill_order = _interior_fill_order(inner_h, inner_w, head_side)
    for idx in range(int(orange_count)):
        rr, cc = fill_order[idx]
        interior[int(rr), int(cc)] = int(ORANGE)

    ir = int(top + thickness)
    ic = int(left + thickness)
    grid[ir : ir + inner_h, ic : ic + inner_w] = interior


def _paint_points(grid: np.ndarray, pts: set[tuple[int, int]], color: int, pr: int = 0, pc: int = 0) -> None:
    for r, c in pts:
        grid[int(pr + r), int(pc + c)] = int(color)


def _build_assembly(spec: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    layout = _layout_from_spec(spec)
    local_h = int(layout["total_h"])
    local_w = int(layout["total_w"])
    input_patch = np.full((local_h, local_w), BG, dtype=int)
    output_patch = np.full((local_h, local_w), BG, dtype=int)

    top = int(layout["cont_r"])
    left = int(layout["cont_c"])
    steps = int(layout["steps"])
    dr, dc = SIDE_TO_DIR[int(layout["head_side"])]

    _paint_points(input_patch, layout["marker_cells_local"], HEAD_COLOR)
    _paint_points(output_patch, layout["marker_cells_local"], HEAD_COLOR)

    _draw_container(
        input_patch,
        top,
        left,
        int(layout["inner_h"]),
        int(layout["inner_w"]),
        int(layout["thickness"]),
        int(layout["head_side"]),
        int(layout["initial_orange"]),
    )
    _paint_points(input_patch, layout["head_cells_local"], HEAD_COLOR)

    _draw_container(
        output_patch,
        int(top + dr * steps),
        int(left + dc * steps),
        int(layout["inner_h"]),
        int(layout["inner_w"]),
        int(layout["thickness"]),
        int(layout["head_side"]),
        int(layout["initial_orange"] + steps),
    )
    moved_head = {(int(r + dr * steps), int(c + dc * steps)) for r, c in layout["head_cells_local"]}
    _paint_points(output_patch, moved_head, HEAD_COLOR)
    return input_patch, output_patch, layout


def _marker_matches_unique(assemblies: list[dict], placements: list[tuple[int, int]]) -> bool:
    absolute = []
    for a, (pr, pc) in zip(assemblies, placements):
        direction = SIDE_TO_DIR[int(a["head_side"])]
        absolute.append(
            {
                "direction": direction,
                "head_anchor": (int(pr + a["head_anchor_local"][0]), int(pc + a["head_anchor_local"][1])),
                "marker_anchor": (int(pr + a["marker_anchor_local"][0]), int(pc + a["marker_anchor_local"][1])),
                "index": len(absolute),
            }
        )

    for entry in absolute:
        direction = entry["direction"]
        hr, hc = entry["head_anchor"]
        matches = []
        for other in absolute:
            ar, ac = other["marker_anchor"]
            if direction[0] != 0:
                if int(ac) != int(hc):
                    continue
                delta = int(ar - hr)
                if int(direction[0]) * delta <= 0 or abs(delta) != MARKER_DISTANCE:
                    continue
            else:
                if int(ar) != int(hr):
                    continue
                delta = int(ac - hc)
                if int(direction[1]) * delta <= 0 or abs(delta) != MARKER_DISTANCE:
                    continue
            matches.append(int(other["index"]))
        if matches != [int(entry["index"])]:
            return False
    return True


def _has_nondefault_L_mover(assemblies: list[dict]) -> bool:
    for a in assemblies:
        if a["is_decoy"] or a["marker_kind"] != "L":
            continue
        hs = int(a["head_side"])
        if hs in (0, 1):
            cells = sorted(a["head_cells_local"], key=lambda p: (p[1], p[0]))
        else:
            cells = sorted(a["head_cells_local"], key=lambda p: (p[0], p[1]))
        if tuple(a["head_anchor_local"]) != tuple(cells[0]):
            return True
    return False


def _gen_attempt(return_meta: bool = False):
    n_containers = random.randint(3, 4)
    n_decoys = random.choice([0, 1])
    n_movers = n_containers - n_decoys
    if n_movers < 2:
        n_decoys = n_containers - 2
        n_movers = 2

    if n_containers == 3:
        max_inner_h, max_inner_w = 4, 4
    else:
        max_inner_h, max_inner_w = 3, 4

    marker_kinds = _choose_marker_kinds(n_containers)
    specs = []
    for i in range(n_containers):
        specs.append(
            _make_spec(
                marker_kind=marker_kinds[i],
                is_decoy=(i >= n_movers),
                max_inner_h=max_inner_h,
                max_inner_w=max_inner_w,
            )
        )

    mover_specs = [s for s in specs if not s["is_decoy"]]
    if {"L", "T"} - {str(s["marker_kind"]) for s in mover_specs}:
        return None
    mover_thicknesses = {int(s["thickness"]) for s in mover_specs}
    if len(mover_thicknesses) < 2:
        return None
    if MAX_THICKNESS not in mover_thicknesses:
        return None
    if not any(int(s["thickness"]) < MAX_THICKNESS and int(s["grey"]) > int(s["thickness"]) for s in mover_specs):
        return None
    if not any(int(s["grey"]) > int(s["thickness"]) for s in mover_specs):
        return None

    assemblies = []
    for spec in specs:
        input_patch, output_patch, layout = _build_assembly(spec)
        if input_patch.shape[0] > 26 or input_patch.shape[1] > 26:
            return None
        assemblies.append({**layout, "input": input_patch, "output": output_patch})

    if not _has_nondefault_L_mover(assemblies):
        return None

    assemblies.sort(key=lambda a: int(a["input"].shape[0] * a["input"].shape[1]), reverse=True)

    total_area = sum(int(a["input"].shape[0] * a["input"].shape[1]) for a in assemblies)
    max_h = max(int(a["input"].shape[0]) for a in assemblies)
    max_w = max(int(a["input"].shape[1]) for a in assemblies)
    lo_h = max(8, max_h + 2)
    lo_w = max(8, max_w + 2)
    hi_h = min(26, max(lo_h, int(total_area ** 0.5) + 8))
    hi_w = min(26, max(lo_w, int(total_area ** 0.5) + 8))
    if lo_h > 26 or lo_w > 26:
        return None

    grid_h = random.randint(lo_h, hi_h)
    grid_w = random.randint(lo_w, hi_w)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    placements: list[tuple[int, int]] = []

    for asm in assemblies:
        h, w = asm["input"].shape
        if h > grid_h or w > grid_w:
            return None
        placed = False
        for _ in range(500):
            r0 = random.randint(0, grid_h - h)
            c0 = random.randint(0, grid_w - w)
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(grid_h, r0 + h + 1)
            cc1 = min(grid_w, c0 + w + 1)
            if not occupied[rr0:rr1, cc0:cc1].any():
                occupied[r0 : r0 + h, c0 : c0 + w] = True
                placements.append((r0, c0))
                placed = True
                break
        if not placed:
            return None

    if not _marker_matches_unique(assemblies, placements):
        return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    expected = np.full((grid_h, grid_w), BG, dtype=int)
    for asm, (r0, c0) in zip(assemblies, placements):
        paste(grid, asm["input"], int(r0), int(c0))
        paste(expected, asm["output"], int(r0), int(c0))

    try:
        output = main(grid)
    except Exception:
        return None
    if not np.array_equal(output, expected):
        return None

    if return_meta:
        return grid, {"assemblies": assemblies, "placements": placements, "expected": expected}
    return grid


def generate_input() -> np.ndarray:
    for _ in range(1200):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid combo variant after 1200 attempts")
