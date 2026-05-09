# Combo variant of 271d71e2 (index 21)
# combo_spec: D1/2+D4/2
# Name: 内部图案决定方向加箭头头部

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, bbox_coords, colors_present, components_mask, paste


# -----------------------------------------------------------------------------
# Design analysis for the fused variant
#
# a) Concrete behavior and trace
# - Core object is still a black rectangular container with grey/orange interior.
# - Outside the container there is now a 3x3 maroon arrow head plus a matching
#   3x3 maroon destination arrow further along the travel direction.
# - The crucial fusion rule is:
#     1. Read movement direction from the internal orange motif.
#     2. Use the external arrow only as the movable head sprite and as a visual
#        confirmation; it does not get to decide direction by attachment side.
# - To make that visible, the head arrow is always attached on a side orthogonal
#   to the true movement axis. Example:
#     * container moves right
#     * internal orange cells form a right-pointing arrow template
#     * external arrow head also points right, but it is attached above or below
#       the container, so "head sits on the right" is false
#     * solver must infer R from the interior, then find the matching maroon
#       destination arrow to the right of the source arrow, then move container
#       and head right by min(distance, grey_count)
# - Compared with the original task:
#     * original: edge-adjacent line position decides direction
#     * fused: internal orange arrow decides direction, head position is decoupled
#       from direction, and the head itself is a compact arrow sprite
#
# b) Complexity calibration
# - Direction is limited to four cardinal orientations.
# - The internal cue is a fixed 4-cell arrow template, not an arbitrary shape.
# - The external cue is a fixed 3x3 arrow sprite.
# - Head attachment is restricted to the two sides orthogonal to motion. This is
#   enough to visibly break the original heuristic without creating too many
#   degrees of freedom.
#
# c) Implicit assumption audit against the original solver
# - Original assumption: direction can be read from which side the head touches.
#   Broken: the head is orthogonally attached, so side != travel direction.
# - Original assumption: the head can be reduced to the border-nearest pixels and
#   still keep its identity. Broken: a 3x3 arrow must be tracked as a full shape.
# - Original assumption: grey fill order depends on head side. Broken: fill order
#   must depend on the internal arrow direction instead.
# - New assumption introduced here: movers have a unique internal arrow template.
#   Generator guarantees that by using exactly the 4 template cells as the input
#   orange set for movers; stationary decoys are all-orange and skipped because
#   grey_count == 0 before direction inference.
# - New assumption introduced here: the external arrow direction agrees with the
#   internal arrow direction. Generator guarantees this and solver checks it.
#
# d) Placement legality
# - Each generated object reserves the union of:
#     input container, input head arrow, destination arrow,
#     moved container, moved head arrow
# - These reserve rectangles are packed with a 1-cell background moat, so
#   different assemblies never overlap in input or output.
# - The reserve box also contains the travel corridor, preventing unrelated
#   arrows from appearing between a source head and its own destination.
#
# e) Visual observability
# - In every mover, the head is visibly not on the side of travel, so the viewer
#   cannot use the original "which side is the head on?" rule.
# - The interior orange motif is small but salient and repeats across examples.
# - The head arrow points the same way as the internal motif, acting as a second
#   visible cue rather than introducing a separate mapping.
#
# f) Visual cleanliness
# - Only five colors appear: 0/5/6/7/9.
# - Each assembly contains one container, one source arrow, one destination arrow.
# - No element overlaps another except the intended final overlap when a moved
#   head lands exactly on its destination arrow.
#
# g) Inferability of change
# - No new color mapping is introduced.
# - Orange growth is structurally tied to the same inferred direction: the arrow
#   template is the prefix of the fill order, and additional orange cells extend
#   inward from that pointing side.
# - Because the head is orthogonally attached in generated movers, the samples
#   materially differ from the original task rather than merely repainting the
#   old same-side head as an arrow.
# -----------------------------------------------------------------------------

BG = 6
BORDER = 0
GREY = 5
ORANGE = 7
HEAD_COLOR = 9

DIR_TO_VEC = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}
ORTH_ATTACH = {
    "U": ("L", "R"),
    "D": ("L", "R"),
    "L": ("U", "D"),
    "R": ("U", "D"),
}
ARROW_SHAPES = {
    "U": {(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)},
    "D": {(0, 1), (1, 1), (2, 0), (2, 1), (2, 2)},
    "L": {(0, 0), (1, 0), (1, 1), (1, 2), (2, 0)},
    "R": {(0, 2), (1, 0), (1, 1), (1, 2), (2, 2)},
}
SHAPE_TO_DIR = {frozenset(shape): direction for direction, shape in ARROW_SHAPES.items()}


def _rect_border_coords(r0: int, c0: int, r1: int, c1: int):
    for c in range(c0, c1 + 1):
        yield (r0, c)
        yield (r1, c)
    for r in range(r0 + 1, r1):
        yield (r, c0)
        yield (r, c1)


def _arrow_template(inner_h: int, inner_w: int, direction: str) -> list[tuple[int, int]]:
    mid_r = inner_h // 2
    mid_c = inner_w // 2
    if direction == "U":
        return [(0, mid_c), (1, mid_c - 1), (1, mid_c), (1, mid_c + 1)]
    if direction == "D":
        return [
            (inner_h - 1, mid_c),
            (inner_h - 2, mid_c - 1),
            (inner_h - 2, mid_c),
            (inner_h - 2, mid_c + 1),
        ]
    if direction == "L":
        return [(mid_r, 0), (mid_r - 1, 1), (mid_r, 1), (mid_r + 1, 1)]
    if direction == "R":
        return [
            (mid_r, inner_w - 1),
            (mid_r - 1, inner_w - 2),
            (mid_r, inner_w - 2),
            (mid_r + 1, inner_w - 2),
        ]
    raise ValueError(direction)


def _fill_order(inner_h: int, inner_w: int, direction: str) -> list[tuple[int, int]]:
    base = _arrow_template(inner_h, inner_w, direction)
    base_set = set(base)
    mid_r = inner_h // 2
    mid_c = inner_w // 2
    remaining: list[tuple[tuple[int, int, int], tuple[int, int]]] = []
    for r in range(inner_h):
        for c in range(inner_w):
            if (r, c) in base_set:
                continue
            if direction == "U":
                key = (r, abs(c - mid_c), c)
            elif direction == "D":
                key = (inner_h - 1 - r, abs(c - mid_c), c)
            elif direction == "L":
                key = (c, abs(r - mid_r), r)
            else:
                key = (inner_w - 1 - c, abs(r - mid_r), r)
            remaining.append((key, (r, c)))
    remaining.sort(key=lambda item: item[0])
    return base + [cell for _key, cell in remaining]


def _infer_inner_direction(inner: np.ndarray) -> str | None:
    orange = {(int(r), int(c)) for r, c in np.argwhere(inner == ORANGE)}
    if not orange:
        return None
    matches = []
    inner_h, inner_w = inner.shape
    for direction in DIR_TO_VEC:
        template = set(_arrow_template(inner_h, inner_w, direction))
        if template.issubset(orange):
            matches.append(direction)
    if len(matches) == 1:
        return matches[0]
    return None


def _arrow_tl_relative(outer_h: int, outer_w: int, attach_side: str, anchor: int) -> tuple[int, int]:
    if attach_side == "U":
        return (-3, anchor)
    if attach_side == "D":
        return (outer_h, anchor)
    if attach_side == "L":
        return (anchor, -3)
    return (anchor, outer_w)


def _stamp_arrow(grid: np.ndarray, tl: tuple[int, int], direction: str, color: int) -> None:
    r0, c0 = tl
    for dr, dc in ARROW_SHAPES[direction]:
        grid[int(r0 + dr), int(c0 + dc)] = int(color)


def _rect_union(rects: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    return (
        min(r0 for r0, _c0, _r1, _c1 in rects),
        min(c0 for _r0, c0, _r1, _c1 in rects),
        max(r1 for _r0, _c0, r1, _c1 in rects),
        max(c1 for _r0, _c0, _r1, c1 in rects),
    )


def _build_container_patch(inner_h: int, inner_w: int, direction: str, orange_count: int) -> np.ndarray:
    patch = np.full((inner_h + 2, inner_w + 2), BORDER, dtype=int)
    inner = np.full((inner_h, inner_w), GREY, dtype=int)
    order = _fill_order(inner_h, inner_w, direction)
    for idx, (r, c) in enumerate(order):
        if idx >= orange_count:
            break
        inner[r, c] = ORANGE
    patch[1:-1, 1:-1] = inner
    return patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fused rule:
    - Read direction from the internal orange arrow template.
    - The adjacent external 3x3 arrow is the movable head sprite, but its
      attachment side does not determine direction.
    - Move container + head toward the nearest matching arrow destination along
      the internally inferred direction, capped by the number of grey cells.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    out = g.copy()

    cnt = Counter(map(int, g.flatten()))
    bg = int(max((c for c in cnt.keys() if int(c) != BORDER), key=lambda c: int(cnt[int(c)])))
    dir4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    containers: list[tuple[int, int, int, int]] = []
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
        containers.append((r0, c0, r1, c1))

    assert containers

    marker_colors = colors_present(g, exclude={bg, BORDER, GREY, ORANGE})
    comp_by_color: dict[int, list[dict]] = defaultdict(list)
    for col in marker_colors:
        labc, nc = label(g == int(col), structure=STRUCT4)
        for k in range(1, int(nc) + 1):
            coords = np.argwhere(labc == int(k))
            if coords.size == 0:
                continue
            rr0, cc0, _rr1, _cc1 = bbox_coords(coords)
            pts = {(int(r), int(c)) for r, c in coords}
            shape = {(int(r - rr0), int(c - cc0)) for r, c in pts}
            comp_by_color[int(col)].append(
                {"pts": pts, "shape": shape, "tl": (int(rr0), int(cc0))}
            )

    def pick_adjacent_head(box_: tuple[int, int, int, int]) -> dict | None:
        r0, c0, r1, c1 = box_
        seeds: list[tuple[int, int, int]] = []
        for r, c in _rect_border_coords(r0, c0, r1, c1):
            for dr, dc in dir4:
                rr, cc = int(r + dr), int(c + dc)
                if 0 <= rr < H and 0 <= cc < W:
                    val = int(g[rr, cc])
                    if val not in (int(bg), int(BORDER), int(GREY), int(ORANGE)):
                        seeds.append((rr, cc, val))
        if not seeds:
            return None

        chosen_color = int(Counter(val for _r, _c, val in seeds).most_common(1)[0][0])
        seed_pts = {(r, c) for r, c, val in seeds if int(val) == chosen_color}
        for comp in comp_by_color.get(chosen_color, []):
            if not (comp["pts"] & seed_pts):
                continue
            head_dir = SHAPE_TO_DIR.get(frozenset(comp["shape"]))
            if head_dir is None:
                continue
            return {
                "color": chosen_color,
                "pts": set(comp["pts"]),
                "shape": comp["shape"],
                "tl": comp["tl"],
                "direction": head_dir,
            }
        return None

    for r0, c0, r1, c1 in containers:
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        grey_count = int(np.sum(inner == GREY))
        if grey_count == 0:
            continue

        direction = _infer_inner_direction(inner)
        if direction is None:
            continue

        head = pick_adjacent_head((r0, c0, r1, c1))
        if head is None or head["direction"] != direction:
            continue

        dr, dc = DIR_TO_VEC[direction]
        best_dist = None
        best_dest = None
        for comp in comp_by_color.get(head["color"], []):
            if comp["pts"] == head["pts"] or comp["shape"] != head["shape"]:
                continue
            tr0 = int(comp["tl"][0] - head["tl"][0])
            tc0 = int(comp["tl"][1] - head["tl"][1])
            if dr != 0:
                if tc0 != 0 or tr0 * dr <= 0:
                    continue
            else:
                if tr0 != 0 or tc0 * dc <= 0:
                    continue
            dist = abs(tr0) + abs(tc0)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_dest = comp
        if best_dest is None or best_dist is None:
            continue

        steps = min(int(best_dist), grey_count)
        if steps <= 0:
            continue

        orange_count = int(np.sum(inner == ORANGE))
        moved_patch = _build_container_patch(inner.shape[0], inner.shape[1], direction, orange_count + steps)

        out[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
        for rr, cc in head["pts"]:
            if int(out[rr, cc]) == int(head["color"]):
                out[rr, cc] = int(bg)

        paste(out, moved_patch, int(r0 + dr * steps), int(c0 + dc * steps), clip=True)
        for rr, cc in head["pts"]:
            nr = int(rr + dr * steps)
            nc = int(cc + dc * steps)
            if 0 <= nr < H and 0 <= nc < W:
                out[nr, nc] = int(head["color"])

    return out


def _build_input_interior(inner_h: int, inner_w: int, direction: str, stationary: bool) -> np.ndarray:
    if stationary:
        return np.full((inner_h, inner_w), ORANGE, dtype=int)
    interior = np.full((inner_h, inner_w), GREY, dtype=int)
    for r, c in _arrow_template(inner_h, inner_w, direction):
        interior[r, c] = ORANGE
    return interior


def _gen_attempt() -> np.ndarray | None:
    n_containers = random.randint(2, 4)
    n_decoys = random.choice([0, 0, 1])
    mover_count = max(1, n_containers - n_decoys)
    n_decoys = n_containers - mover_count

    directions = [random.choice(list(DIR_TO_VEC.keys())) for _ in range(n_containers)]
    if mover_count >= 2 and len(set(directions[:mover_count])) == 1:
        directions[1] = random.choice([d for d in DIR_TO_VEC if d != directions[0]])

    specs: list[dict] = []
    for i in range(n_containers):
        stationary = i >= mover_count
        direction = directions[i]
        inner_h = random.randint(3, 7)
        inner_w = random.randint(3, 5)
        outer_h = inner_h + 2
        outer_w = inner_w + 2
        inner_area = inner_h * inner_w
        attach_side = random.choice(ORTH_ATTACH[direction])
        gap = random.randint(0, 6)
        travel = 4 + gap

        if stationary:
            initial_orange = inner_area
            steps = 0
        else:
            if inner_area <= 4:
                return None
            initial_orange = 4
            steps = min(inner_area - initial_orange, travel)
            if steps <= 0:
                return None

        if attach_side in ("U", "D"):
            anchor = random.randint(0, outer_w - 3)
        else:
            anchor = random.randint(0, outer_h - 3)

        head_r, head_c = _arrow_tl_relative(outer_h, outer_w, attach_side, anchor)
        dr, dc = DIR_TO_VEC[direction]
        dest_r = head_r + dr * travel
        dest_c = head_c + dc * travel
        out_cont_r = dr * steps
        out_cont_c = dc * steps
        out_head_r = head_r + dr * steps
        out_head_c = head_c + dc * steps

        rects = [
            (0, 0, outer_h - 1, outer_w - 1),
            (head_r, head_c, head_r + 2, head_c + 2),
            (dest_r, dest_c, dest_r + 2, dest_c + 2),
            (out_cont_r, out_cont_c, out_cont_r + outer_h - 1, out_cont_c + outer_w - 1),
            (out_head_r, out_head_c, out_head_r + 2, out_head_c + 2),
        ]
        min_r, min_c, max_r, max_c = _rect_union(rects)
        shift_r = -min_r
        shift_c = -min_c

        specs.append(
            {
                "direction": direction,
                "attach_side": attach_side,
                "inner_h": inner_h,
                "inner_w": inner_w,
                "outer_h": outer_h,
                "outer_w": outer_w,
                "initial_orange": initial_orange,
                "steps": steps,
                "cont_tl": (shift_r, shift_c),
                "head_tl": (head_r + shift_r, head_c + shift_c),
                "dest_tl": (dest_r + shift_r, dest_c + shift_c),
                "out_cont_tl": (out_cont_r + shift_r, out_cont_c + shift_c),
                "out_head_tl": (out_head_r + shift_r, out_head_c + shift_c),
                "reserve_h": max_r - min_r + 1,
                "reserve_w": max_c - min_c + 1,
                "interior": _build_input_interior(inner_h, inner_w, direction, stationary),
            }
        )

    total_area = sum(spec["reserve_h"] * spec["reserve_w"] for spec in specs)
    max_res_h = max(spec["reserve_h"] for spec in specs)
    max_res_w = max(spec["reserve_w"] for spec in specs)
    base = max(10, int(total_area ** 0.5) + 4)
    lo_h = max(10, max_res_h + 2, min(26, base))
    lo_w = max(10, max_res_w + 2, min(26, base))
    hi_h = min(26, max(lo_h, base + 7))
    hi_w = min(26, max(lo_w, base + 7))
    if lo_h > 26 or lo_w > 26:
        return None

    grid_h = random.randint(lo_h, hi_h)
    grid_w = random.randint(lo_w, hi_w)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    order = sorted(
        range(len(specs)),
        key=lambda idx: specs[idx]["reserve_h"] * specs[idx]["reserve_w"],
        reverse=True,
    )
    placements: dict[int, tuple[int, int]] = {}
    for idx in order:
        spec = specs[idx]
        rh = spec["reserve_h"]
        rw = spec["reserve_w"]
        if rh > grid_h or rw > grid_w:
            return None
        placed = False
        for _ in range(400):
            r0 = random.randint(0, grid_h - rh)
            c0 = random.randint(0, grid_w - rw)
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(grid_h, r0 + rh + 1)
            cc1 = min(grid_w, c0 + rw + 1)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            occupied[r0 : r0 + rh, c0 : c0 + rw] = True
            placements[idx] = (r0, c0)
            placed = True
            break
        if not placed:
            return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    for idx, spec in enumerate(specs):
        pr, pc = placements[idx]
        cr = pr + spec["cont_tl"][0]
        cc = pc + spec["cont_tl"][1]
        hr = pr + spec["head_tl"][0]
        hc = pc + spec["head_tl"][1]
        dr = pr + spec["dest_tl"][0]
        dc = pc + spec["dest_tl"][1]

        oh = spec["outer_h"]
        ow = spec["outer_w"]
        grid[cr, cc : cc + ow] = BORDER
        grid[cr + oh - 1, cc : cc + ow] = BORDER
        grid[cr : cr + oh, cc] = BORDER
        grid[cr : cr + oh, cc + ow - 1] = BORDER
        grid[cr + 1 : cr + 1 + spec["inner_h"], cc + 1 : cc + 1 + spec["inner_w"]] = spec["interior"]
        _stamp_arrow(grid, (hr, hc), spec["direction"], HEAD_COLOR)
        _stamp_arrow(grid, (dr, dc), spec["direction"], HEAD_COLOR)

    try:
        output = main(grid)
    except Exception:
        return None

    expected = np.array(grid, copy=True)
    for idx, spec in enumerate(specs):
        if spec["steps"] <= 0:
            continue
        pr, pc = placements[idx]
        cr = pr + spec["cont_tl"][0]
        cc = pc + spec["cont_tl"][1]
        hr = pr + spec["head_tl"][0]
        hc = pc + spec["head_tl"][1]
        ncr = pr + spec["out_cont_tl"][0]
        ncc = pc + spec["out_cont_tl"][1]
        nhr = pr + spec["out_head_tl"][0]
        nhc = pc + spec["out_head_tl"][1]

        oh = spec["outer_h"]
        ow = spec["outer_w"]
        expected[cr : cr + oh, cc : cc + ow] = BG
        for rr, cc2 in ARROW_SHAPES[spec["direction"]]:
            expected[hr + rr, hc + cc2] = BG

        patch = _build_container_patch(
            spec["inner_h"],
            spec["inner_w"],
            spec["direction"],
            spec["initial_orange"] + spec["steps"],
        )
        expected[ncr : ncr + oh, ncc : ncc + ow] = patch
        _stamp_arrow(expected, (nhr, nhc), spec["direction"], HEAD_COLOR)

    if not np.array_equal(output, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("Failed to generate fused variant after 400 attempts")
