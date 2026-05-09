# Variant of 271d71e2
# Dimension: D15 — gap=0时的头-目标关系 (meta)
# Change: gap=0时头部和目标线相邻接触（形成连通分量），head_dest_distance=1... → gap=0时头与目标重叠

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste


# Design analysis for the D15 variant
# ===================================
# a) Concrete behavior
#    - Original task: when gap=0 there are still two visible same-color lines, one
#      adjacent to the container (the head) and one immediately beyond it (the
#      destination). The input therefore contains a 2-cell-thick bar in that
#      direction, and head_dest_distance = 1.
#    - Variant: when gap=0 the head and destination collapse onto the same pixels.
#      The input now contains only a single 1-cell-thick line adjacent to the
#      container. Despite geometric distance 0, the container still moves 1 step
#      if it has at least one grey cell. The moved container occupies the old
#      overlapped target position, so the stationary destination is usually
#      occluded by the moved black border.
#    - Small trace (top-headed container):
#      original gap=0 input rows: [dest][head][container]
#      variant gap=0 input rows:  [head=dest][container]
#      original logic would compute steps=min(1, grey), while the new logic must
#      compute steps=1 whenever grey>0 even though no separate destination line
#      exists in the grid.
# b) Complexity calibration
#    - Only the gap=0 case changes; other gaps still behave exactly like the base
#      task. Samples therefore mix overlap movers and ordinary movers so that the
#      exception is visually supported instead of being a one-off edge case.
#    - Containers keep the original rectangular structure, one-color head/dest
#      lines, and the same orange-filling order. This preserves recognizability
#      while making the D15 change visible as “single line but still moves once”.
# c) Hidden assumptions in the original solver that break here
#    - Original code assumes head direction can be read from a head component bbox
#      lying strictly outside the container. That is harmless when head and dest
#      are separate or merely touching, but with complete overlap there is only one
#      visible line and no second destination component to match against.
#    - Original destination search depends on finding another component, or a
#      residual piece of the same component after removing nearest pixels. With
#      perfect overlap there is no residual destination at all.
#    - Original steps = manhattan(head_tl, dest_tl). For overlap this becomes 0,
#      which would incorrectly suppress movement unless a dedicated branch exists.
# d) Placement legality
#    - The generator reserves the union of input and output footprints for each
#      assembly, not only the input footprint. This is necessary because in the
#      overlap case the output extends one step farther than the visible input
#      marker range and the moved container can cover the old marker position.
#    - A 1-cell background moat is kept between these reserved footprints so the
#      solver can safely scan along an assembly axis without colliding with another
#      object's line first.
# e) Visual observability
#    - Every generated puzzle contains at least one mover with gap=0 overlap.
#      When there are multiple movers, at least one ordinary gap>0 mover is kept
#      as contrast. A human can then compare “single visible line yet still moves”
#      against the standard “separate line at some distance”.
# f) Visual cleanliness
#    - Overlap does not add new colors or local clutter. It removes one visible
#      line from the input, which is a clean, legible difference.
# g) Color-change inferability
#    - No new color semantics are introduced. Grey-to-orange charging still follows
#      the original head-facing order, so only the trigger for one-step movement in
#      the overlap case changes.
# h) New solver assumptions after modification
#    - Head detection now assumes the head is the unique full-length monochrome line
#      immediately adjacent to one container edge.
#    - Destination detection assumes any separate destination, if present, lies on
#      the same axis, has the same full span, and the path before it is background.
#    - The generator enforces both conditions by construction.


BG = 6
BORDER = 0
GREY = 5
ORANGE = 7
HEAD_COLOR = 9

# head_side: 0=top, 1=bottom, 2=left, 3=right
SIDE_TO_DIR = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}


def _rect_border_coords(r0: int, c0: int, r1: int, c1: int):
    for c in range(c0, c1 + 1):
        yield (r0, c)
        yield (r1, c)
    for r in range(r0 + 1, r1):
        yield (r, c0)
        yield (r, c1)


def _line_coords_from_box(box: tuple[int, int, int, int], side: int, offset: int = 1) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    k = int(offset)
    if side == 0:
        return [(r0 - k, c) for c in range(c0, c1 + 1)]
    if side == 1:
        return [(r1 + k, c) for c in range(c0, c1 + 1)]
    if side == 2:
        return [(r, c0 - k) for r in range(r0, r1 + 1)]
    return [(r, c1 + k) for r in range(r0, r1 + 1)]


def _in_bounds(shape: tuple[int, int], coords: list[tuple[int, int]]) -> bool:
    h, w = shape
    return all(0 <= int(r) < h and 0 <= int(c) < w for r, c in coords)


def _uniform_line_color(g: np.ndarray, coords: list[tuple[int, int]], excluded: set[int]) -> int | None:
    vals = [int(g[r, c]) for r, c in coords]
    if len(set(vals)) != 1:
        return None
    col = int(vals[0])
    if col in excluded:
        return None
    return col


def _find_containers(g: np.ndarray) -> list[dict]:
    h, w = g.shape
    containers = []
    for pts in components_mask((g == GREY) | (g == ORANGE), connectivity=4):
        if pts.size == 0:
            continue
        ir0, ic0, ir1, ic1 = bbox_coords(pts)
        r0, c0, r1, c1 = int(ir0 - 1), int(ic0 - 1), int(ir1 + 1), int(ic1 + 1)
        if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
            continue
        if any(int(g[r, c]) != BORDER for r, c in _rect_border_coords(r0, c0, r1, c1)):
            continue
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        if inner.size == 0 or not bool(np.isin(inner, [GREY, ORANGE]).all()):
            continue
        containers.append({"box": (r0, c0, r1, c1)})
    return containers


def _detect_head(g: np.ndarray, box: tuple[int, int, int, int], bg: int) -> dict | None:
    excluded = {int(bg), BORDER, GREY, ORANGE}
    candidates = []
    for side in range(4):
        coords = _line_coords_from_box(box, side, offset=1)
        if not _in_bounds(g.shape, coords):
            continue
        col = _uniform_line_color(g, coords, excluded)
        if col is None:
            continue
        rs = [r for r, _c in coords]
        cs = [c for _r, c in coords]
        candidates.append(
            {
                "side": side,
                "dir": SIDE_TO_DIR[side],
                "color": int(col),
                "head_pts": set(coords),
                "head_tl": (int(min(rs)), int(min(cs))),
            }
        )
    if len(candidates) != 1:
        return None
    return candidates[0]


def _find_dest_distance(g: np.ndarray, box: tuple[int, int, int, int], side: int, color: int, bg: int) -> int:
    h, w = g.shape
    limit = max(h, w)
    for dist in range(1, limit + 1):
        coords = _line_coords_from_box(box, side, offset=1 + dist)
        if not _in_bounds(g.shape, coords):
            break
        vals = [int(g[r, c]) for r, c in coords]
        if all(v == int(color) for v in vals):
            return int(dist)
        if any(v != int(bg) for v in vals):
            break
    return 0


def _interior_fill_order(inner_h: int, inner_w: int, head_side: int) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(inner_h) for c in range(inner_w)]
    if head_side == 3:  # right
        cells.sort(key=lambda p: (-p[1], p[0]))
    elif head_side == 2:  # left
        cells.sort(key=lambda p: (p[1], -p[0]))
    elif head_side == 1:  # bottom
        cells.sort(key=lambda p: (-p[0], -p[1]))
    else:  # top
        cells.sort(key=lambda p: (p[0], p[1]))
    return cells


def _grey_fill_subset(
    box: tuple[int, int, int, int], inner: np.ndarray, head_side: int, steps: int
) -> set[tuple[int, int]]:
    r0, c0, _r1, _c1 = box
    grey_coords = [
        (int(r0 + 1 + rr), int(c0 + 1 + cc))
        for rr in range(inner.shape[0])
        for cc in range(inner.shape[1])
        if int(inner[rr, cc]) == GREY
    ]
    order = _interior_fill_order(inner.shape[0], inner.shape[1], head_side)
    local_greys = {
        (int(r - (r0 + 1)), int(c - (c0 + 1))): (int(r), int(c))
        for r, c in grey_coords
    }
    ordered_global = [local_greys[p] for p in order if p in local_greys]
    return set(ordered_global[: int(steps)])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move bordered containers toward same-colored destination markers, charging grey cells to orange.

    D15 variant:
    - If there is no separate destination line beyond the adjacent head line, interpret that
      as gap=0 overlap (head and destination share the same pixels).
    - In that overlap case, steps = 1 whenever at least one grey cell exists.
    """
    g = np.array(grid, dtype=int)
    h, w = g.shape

    cnt = Counter(map(int, g.flatten()))
    bg = int(max((c for c in cnt.keys() if int(c) != BORDER), key=lambda c: int(cnt[int(c)])))

    containers = _find_containers(g)
    assert containers

    out = g.copy()

    for cont in containers:
        box = cont["box"]
        r0, c0, r1, c1 = box

        head = _detect_head(g, box, bg)
        if head is None:
            continue

        head_side = int(head["side"])
        direction = head["dir"]
        hc = int(head["color"])
        head_pts = set(head["head_pts"])

        dest_distance = _find_dest_distance(g, box, head_side, hc, bg)
        overlap_case = dest_distance == 0

        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        grey_count = int(np.count_nonzero(inner == GREY))
        if overlap_case:
            steps = 1 if grey_count > 0 else 0
        else:
            steps = min(int(dest_distance), int(grey_count))

        dr, dc = direction
        tr = int(dr * steps)
        tc = int(dc * steps)
        to_fill = _grey_fill_subset(box, inner, head_side, steps)

        out[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
        if not overlap_case:
            for r, c in head_pts:
                if int(out[r, c]) == hc:
                    out[r, c] = int(bg)

        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        for rr, cc in to_fill:
            pr = int(rr - r0)
            pc = int(cc - c0)
            if int(patch[pr, pc]) == GREY:
                patch[pr, pc] = ORANGE
        paste(out, patch, int(r0 + tr), int(c0 + tc), clip=True)

        for r, c in head_pts:
            rr = int(r + tr)
            cc = int(c + tc)
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = hc

    return out


def _line_coords_for_container_tl(
    container_tl: tuple[int, int], outer_h: int, outer_w: int, head_side: int, offset: int
) -> list[tuple[int, int]]:
    r0, c0 = container_tl
    return _line_coords_from_box((r0, c0, r0 + outer_h - 1, c0 + outer_w - 1), head_side, offset=offset)


def _compute_layout(spec: dict) -> dict:
    outer_h = int(spec["outer_h"])
    outer_w = int(spec["outer_w"])
    head_side = int(spec["head_side"])
    gap = int(spec["gap"])
    steps = int(spec["steps"])
    dr, dc = SIDE_TO_DIR[head_side]

    in_cont = (0, 0)
    out_cont = (int(dr * steps), int(dc * steps))
    dest_distance = 0 if gap == 0 else gap + 1

    in_head = _line_coords_for_container_tl(in_cont, outer_h, outer_w, head_side, offset=1)
    if dest_distance == 0:
        dest = list(in_head)
    else:
        dest = _line_coords_for_container_tl(in_cont, outer_h, outer_w, head_side, offset=1 + dest_distance)
    out_head = [(int(r + dr * steps), int(c + dc * steps)) for r, c in in_head]

    rows = [
        0,
        outer_h - 1,
        out_cont[0],
        out_cont[0] + outer_h - 1,
        *(r for r, _c in in_head),
        *(r for r, _c in dest),
        *(r for r, _c in out_head),
    ]
    cols = [
        0,
        outer_w - 1,
        out_cont[1],
        out_cont[1] + outer_w - 1,
        *(c for _r, c in in_head),
        *(c for _r, c in dest),
        *(c for _r, c in out_head),
    ]
    min_r = int(min(rows))
    max_r = int(max(rows))
    min_c = int(min(cols))
    max_c = int(max(cols))
    shift_r = -min_r
    shift_c = -min_c

    def shift_coords(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
        return [(int(r + shift_r), int(c + shift_c)) for r, c in coords]

    return {
        **spec,
        "total_h": int(max_r - min_r + 1),
        "total_w": int(max_c - min_c + 1),
        "input_container_tl": (int(in_cont[0] + shift_r), int(in_cont[1] + shift_c)),
        "output_container_tl": (int(out_cont[0] + shift_r), int(out_cont[1] + shift_c)),
        "input_head": shift_coords(in_head),
        "dest": shift_coords(dest),
        "output_head": shift_coords(out_head),
    }


def _draw_container(
    grid: np.ndarray,
    tl: tuple[int, int],
    inner_h: int,
    inner_w: int,
    head_side: int,
    orange_count: int,
) -> None:
    r0, c0 = tl
    outer_h = inner_h + 2
    outer_w = inner_w + 2

    grid[r0, c0 : c0 + outer_w] = BORDER
    grid[r0 + outer_h - 1, c0 : c0 + outer_w] = BORDER
    grid[r0 : r0 + outer_h, c0] = BORDER
    grid[r0 : r0 + outer_h, c0 + outer_w - 1] = BORDER

    interior = np.full((inner_h, inner_w), GREY, dtype=int)
    fill_order = _interior_fill_order(inner_h, inner_w, head_side)
    for idx in range(int(orange_count)):
        rr, cc = fill_order[idx]
        interior[rr, cc] = ORANGE
    grid[r0 + 1 : r0 + 1 + inner_h, c0 + 1 : c0 + 1 + inner_w] = interior


def _draw_line(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> None:
    for r, c in coords:
        grid[int(r), int(c)] = int(color)


def _gen_attempt() -> np.ndarray | None:
    n_containers = random.randint(2, 4)
    n_decoys = random.choice([0, 0, 1])
    n_movers = max(1, n_containers - n_decoys)
    n_decoys = n_containers - n_movers

    mover_overlap_count = 1 if n_movers == 1 else random.randint(1, n_movers - 1)
    mover_kinds = ["overlap"] * mover_overlap_count + ["separate"] * (n_movers - mover_overlap_count)
    random.shuffle(mover_kinds)

    specs = []
    for i in range(n_containers):
        is_decoy = i >= n_movers
        inner_h = random.randint(1, 8)
        inner_w = random.randint(1, 5)
        inner_area = inner_h * inner_w
        head_side = random.randint(0, 3)
        outer_h = inner_h + 2
        outer_w = inner_w + 2

        if is_decoy:
            grey = 0
            initial_orange = inner_area
            gap = random.randint(1, 6)
            steps = 0
            mover_type = "decoy"
        else:
            grey = random.randint(1, inner_area)
            initial_orange = inner_area - grey
            mover_type = mover_kinds[i]
            if mover_type == "overlap":
                gap = 0
                steps = 1
            else:
                gap = random.randint(1, 10)
                steps = min(grey, gap + 1)

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
                "mover_type": mover_type,
            }
        )

    layouts = [_compute_layout(spec) for spec in specs]

    total_area = sum(a["total_h"] * a["total_w"] for a in layouts)
    min_side = max(8, int(total_area**0.5) + 2)
    max_side = min(26, min_side + 8)
    max_h = max(a["total_h"] for a in layouts)
    max_w = max(a["total_w"] for a in layouts)

    lo_h = max(8, max_h + 2)
    hi_h = max(lo_h, min(26, max_side + 4))
    lo_w = max(8, max_w + 2)
    hi_w = max(lo_w, min(26, max_side + 4))
    if lo_h > 26 or lo_w > 26:
        return None

    grid_h = random.randint(lo_h, min(26, hi_h))
    grid_w = random.randint(lo_w, min(26, hi_w))

    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    placements: list[tuple[int, int]] = []
    for a in layouts:
        th = int(a["total_h"])
        tw = int(a["total_w"])
        if th > grid_h or tw > grid_w:
            return None
        placed = False
        for _ in range(400):
            r0 = random.randint(0, grid_h - th)
            c0 = random.randint(0, grid_w - tw)
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(grid_h, r0 + th + 1)
            cc1 = min(grid_w, c0 + tw + 1)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            occupied[r0 : r0 + th, c0 : c0 + tw] = True
            placements.append((r0, c0))
            placed = True
            break
        if not placed:
            return None

    inp = np.full((grid_h, grid_w), BG, dtype=int)
    expected = np.full((grid_h, grid_w), BG, dtype=int)

    for a, (pr, pc) in zip(layouts, placements):
        input_tl = (int(pr + a["input_container_tl"][0]), int(pc + a["input_container_tl"][1]))
        output_tl = (int(pr + a["output_container_tl"][0]), int(pc + a["output_container_tl"][1]))

        input_head = [(int(pr + r), int(pc + c)) for r, c in a["input_head"]]
        dest = [(int(pr + r), int(pc + c)) for r, c in a["dest"]]
        output_head = [(int(pr + r), int(pc + c)) for r, c in a["output_head"]]

        _draw_container(inp, input_tl, a["inner_h"], a["inner_w"], a["head_side"], a["initial_orange"])
        _draw_line(inp, input_head, HEAD_COLOR)
        _draw_line(inp, dest, HEAD_COLOR)

        _draw_line(expected, dest, HEAD_COLOR)
        _draw_container(
            expected,
            output_tl,
            a["inner_h"],
            a["inner_w"],
            a["head_side"],
            a["initial_orange"] + a["steps"],
        )
        _draw_line(expected, output_head, HEAD_COLOR)

    try:
        output = main(inp)
    except Exception:
        return None
    if not np.array_equal(output, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 300 attempts")


def _variant_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.array(grid, dtype=int)
    cnt = Counter(map(int, g.flatten()))
    bg = int(max((c for c in cnt.keys() if int(c) != BORDER), key=lambda c: int(cnt[int(c)])))
    overlap_movers = 0
    separate_movers = 0
    decoys = 0
    for cont in _find_containers(g):
        box = cont["box"]
        head = _detect_head(g, box, bg)
        if head is None:
            continue
        inner = g[box[0] + 1 : box[2], box[1] + 1 : box[3]]
        grey_count = int(np.count_nonzero(inner == GREY))
        dist = _find_dest_distance(g, box, int(head["side"]), int(head["color"]), bg)
        if grey_count == 0:
            decoys += 1
        elif dist == 0:
            overlap_movers += 1
        else:
            separate_movers += 1
    return {
        "overlap_movers": int(overlap_movers),
        "separate_movers": int(separate_movers),
        "decoys": int(decoys),
    }
