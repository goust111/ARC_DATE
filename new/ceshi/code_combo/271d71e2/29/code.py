# Combo variant of 271d71e2 (index 29)
# combo_spec: D13/0+D15/0
# Name: 取max步数加重叠gap

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste


# Design analysis for the combined variant
# ========================================
# a) Concrete behavior and trace
#    The visual objects stay the same: magenta background, black rectangular containers
#    with grey/orange interiors, one adjacent maroon head line, and one same-color
#    destination line on that axis. The fusion changes two linked pieces:
#      - movement steps = max(distance, grey_count)
#      - when gap=0, head and destination fully overlap, so distance=0
#    Small top-headed trace:
#      base task, gap=0, grey=3:
#        two visible lines above the box, distance=1, steps=min(1,3)=1.
#      fused task, gap=0, grey=3:
#        only one visible line above the box because head=dest, distance=0,
#        steps=max(0,3)=3. The box moves 3 cells upward and all 3 greys turn orange.
#      separate-destination trace, gap=4 => distance=5, grey=2:
#        base task: steps=min(5,2)=2, box stops short.
#        fused task: steps=max(5,2)=5, box reaches the destination line and all greys fill.
#      overshoot trace, gap=2 => distance=3, grey=5:
#        base task: steps=3, some grey remains.
#        fused task: steps=5, the box passes beyond the destination and ends fully orange.
#
# b) Complexity calibration
#    The generator always mixes three observable branches:
#      - one overlap mover (distance=0, still moves by grey_count),
#      - one distance-dominant mover (distance > grey),
#      - one grey-dominant mover (grey > distance).
#    This prevents degeneration into cases where max and min accidentally agree.
#    Container sizes and movement lengths are kept moderate so 3-4 assemblies fit cleanly.
#
# c) Hidden assumptions broken from the original solver
#    - The original component-based head/destination split assumes a separate destination
#      component or at least a residual same-color piece. Full overlap removes that.
#    - The original `steps` variable couples translation length and fill count, which only
#      works when steps <= grey_count. Under `max`, translation can exceed grey_count.
#    - The original generator treated fully-orange boxes as stationary decoys; under max,
#      a separate destination would still move them. To keep the rule legible, this combo
#      generator avoids fully-orange movers entirely.
#
# d) Placement legality
#    Each assembly reserves the full union of:
#      input container, input head, static destination, output container, output head.
#    Assemblies are placed with a 1-cell background moat, so scanning outward from one
#    container never encounters another assembly before its own destination corridor ends.
#
# e) Visual observability
#    A human sees three recurring effects in the same puzzle family:
#      - a single visible adjacent line can still cause a multi-cell move (overlap case),
#      - some boxes move farther than their remaining grey count would allow in the base task,
#      - some boxes overshoot the destination because grey_count is larger than distance.
#    Those three branches collectively support the intended "use max, not min" inference.
#
# f) Visual cleanliness
#    No new symbols or colors are added. All objects remain axis-aligned rectangles/lines.
#    The reserved swept frame prevents unrelated boxes from overlapping a moved corridor.
#
# g) Color-change inferability
#    Grey still means available charge and orange still means consumed charge. Because
#    steps >= grey_count in every mover branch here, all grey cells end orange after moving.
#    The important visible variable is therefore movement length relative to the marker line.
#
# h) New solver assumptions after modification
#    - The head is the unique full-length monochrome line adjacent to one border edge.
#    - If scanning farther along that axis finds no separate line before any non-background
#      blockage, the instance is treated as overlap with distance=0.
#    - The generator enforces a clear background corridor, unique adjacent head, and
#      aligned full-span destination lines whenever distance>0.


BG = 6
BORDER = 0
GREY = 5
ORANGE = 7
HEAD_COLOR = 9

SIDE_TO_DIR = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

MAX_STEPS = 7
MAX_DISTANCE = 6


def _rect_border_coords(r0: int, c0: int, r1: int, c1: int):
    for c in range(c0, c1 + 1):
        yield (r0, c)
        yield (r1, c)
    for r in range(r0 + 1, r1):
        yield (r, c0)
        yield (r, c1)


def _line_coords_from_box(box: tuple[int, int, int, int], side: int, offset: int = 1) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    if side == 0:
        return [(r0 - offset, c) for c in range(c0, c1 + 1)]
    if side == 1:
        return [(r1 + offset, c) for c in range(c0, c1 + 1)]
    if side == 2:
        return [(r, c0 - offset) for r in range(r0, r1 + 1)]
    return [(r, c1 + offset) for r in range(r0, r1 + 1)]


def _in_bounds(shape: tuple[int, int], coords: list[tuple[int, int]]) -> bool:
    h, w = shape
    return all(0 <= int(r) < h and 0 <= int(c) < w for r, c in coords)


def _uniform_line_color(g: np.ndarray, coords: list[tuple[int, int]], excluded: set[int]) -> int | None:
    vals = [int(g[r, c]) for r, c in coords]
    if len(set(vals)) != 1:
        return None
    color = int(vals[0])
    if color in excluded:
        return None
    return color


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
        color = _uniform_line_color(g, coords, excluded)
        if color is None:
            continue
        rs = [r for r, _c in coords]
        cs = [c for _r, c in coords]
        candidates.append(
            {
                "side": int(side),
                "dir": SIDE_TO_DIR[side],
                "color": int(color),
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
    for distance in range(1, limit + 1):
        coords = _line_coords_from_box(box, side, offset=1 + distance)
        if not _in_bounds(g.shape, coords):
            break
        vals = [int(g[r, c]) for r, c in coords]
        if all(v == int(color) for v in vals):
            return int(distance)
        if any(v != int(bg) for v in vals):
            break
    return 0


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


def _grey_fill_subset(
    box: tuple[int, int, int, int], inner: np.ndarray, head_side: int, fill_count: int
) -> set[tuple[int, int]]:
    r0, c0, _r1, _c1 = box
    local_order = _interior_fill_order(inner.shape[0], inner.shape[1], head_side)
    local_greys = {
        (int(rr), int(cc)): (int(r0 + 1 + rr), int(c0 + 1 + cc))
        for rr in range(inner.shape[0])
        for cc in range(inner.shape[1])
        if int(inner[rr, cc]) == GREY
    }
    ordered_global = [local_greys[p] for p in local_order if p in local_greys]
    return set(ordered_global[: int(fill_count)])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move containers toward their same-colored line markers.

    Combined rule:
    - distance = 0 when head and destination overlap (gap=0), otherwise the axis offset
      from head line to destination line.
    - movement_steps = max(distance, grey_count)
    - only existing grey cells can turn orange, so fill_count = grey_count
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
        color = int(head["color"])
        head_pts = set(head["head_pts"])

        distance = _find_dest_distance(g, box, head_side, color, bg)
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        grey_count = int(np.count_nonzero(inner == GREY))

        movement_steps = max(int(distance), int(grey_count))
        # Under max-steps, movement can exceed available grey cells, but recoloring cannot.
        fill_count = min(int(movement_steps), int(grey_count))

        dr, dc = direction
        tr = int(dr * movement_steps)
        tc = int(dc * movement_steps)
        to_fill = _grey_fill_subset(box, inner, head_side, fill_count)

        out[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
        if distance > 0:
            for r, c in head_pts:
                if int(out[r, c]) == int(color):
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
                out[rr, cc] = int(color)

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
    distance = int(spec["distance"])
    steps = int(spec["steps"])
    dr, dc = SIDE_TO_DIR[head_side]

    input_cont = (0, 0)
    output_cont = (int(dr * steps), int(dc * steps))
    input_head = _line_coords_for_container_tl(input_cont, outer_h, outer_w, head_side, offset=1)
    if distance == 0:
        dest = list(input_head)
    else:
        dest = _line_coords_for_container_tl(input_cont, outer_h, outer_w, head_side, offset=1 + distance)
    output_head = [(int(r + dr * steps), int(c + dc * steps)) for r, c in input_head]

    rows = [
        0,
        outer_h - 1,
        output_cont[0],
        output_cont[0] + outer_h - 1,
        *(r for r, _c in input_head),
        *(r for r, _c in dest),
        *(r for r, _c in output_head),
    ]
    cols = [
        0,
        outer_w - 1,
        output_cont[1],
        output_cont[1] + outer_w - 1,
        *(c for _r, c in input_head),
        *(c for _r, c in dest),
        *(c for _r, c in output_head),
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
        "input_container_tl": (int(input_cont[0] + shift_r), int(input_cont[1] + shift_c)),
        "output_container_tl": (int(output_cont[0] + shift_r), int(output_cont[1] + shift_c)),
        "input_head": shift_coords(input_head),
        "dest": shift_coords(dest),
        "output_head": shift_coords(output_head),
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
    for idx in range(min(int(orange_count), inner_h * inner_w)):
        rr, cc = fill_order[idx]
        interior[rr, cc] = ORANGE
    grid[r0 + 1 : r0 + 1 + inner_h, c0 + 1 : c0 + 1 + inner_w] = interior


def _draw_line(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> None:
    for r, c in coords:
        grid[int(r), int(c)] = int(color)


def _base_branches(n_containers: int) -> list[str]:
    branches = ["overlap", "distance_dominant", "grey_dominant"]
    extras = ["overlap", "distance_dominant", "grey_dominant", "equal"]
    while len(branches) < n_containers:
        branches.append(random.choice(extras))
    random.shuffle(branches)
    return branches


def _make_spec(branch: str) -> dict | None:
    for _ in range(300):
        inner_h = random.randint(1, 5)
        inner_w = random.randint(1, 4)
        inner_area = inner_h * inner_w
        outer_h = inner_h + 2
        outer_w = inner_w + 2
        head_side = random.randint(0, 3)

        max_grey = min(inner_area, MAX_STEPS)
        if max_grey < 1:
            continue

        if branch == "overlap":
            grey = random.randint(1, max_grey)
            distance = 0
        elif branch == "distance_dominant":
            distance = random.randint(2, MAX_DISTANCE)
            grey = random.randint(1, min(max_grey, distance - 1))
        elif branch == "grey_dominant":
            if max_grey < 2:
                continue
            grey = random.randint(2, max_grey)
            distance = random.randint(1, grey - 1)
        else:
            grey = random.randint(1, max_grey)
            distance = grey

        steps = max(int(distance), int(grey))
        if steps > MAX_STEPS:
            continue

        axis_extent = (outer_h if head_side in (0, 1) else outer_w) + steps + max(distance, 1)
        if axis_extent > 18:
            continue

        return {
            "branch": branch,
            "inner_h": int(inner_h),
            "inner_w": int(inner_w),
            "outer_h": int(outer_h),
            "outer_w": int(outer_w),
            "head_side": int(head_side),
            "distance": int(distance),
            "gap": 0 if int(distance) == 0 else int(distance) - 1,
            "grey": int(grey),
            "steps": int(steps),
            "initial_orange": int(inner_area - grey),
        }
    return None


def _gen_attempt() -> np.ndarray | None:
    # Use 3-4 assemblies so every sample can simultaneously exhibit overlap,
    # distance-dominant, and grey-dominant behavior.
    n_containers = random.randint(3, 4)
    branches = _base_branches(n_containers)

    layouts = []
    for branch in branches:
        spec = _make_spec(branch)
        if spec is None:
            return None
        layouts.append(_compute_layout(spec))

    total_area = sum(a["total_h"] * a["total_w"] for a in layouts)
    min_side = max(10, int(total_area**0.5) + 3)
    max_side = min(26, min_side + 6)
    max_h = max(a["total_h"] for a in layouts)
    max_w = max(a["total_w"] for a in layouts)

    lo_h = max(10, max_h + 2)
    lo_w = max(10, max_w + 2)
    hi_h = max(lo_h, min(26, max_side + 2))
    hi_w = max(lo_w, min(26, max_side + 2))
    if lo_h > 26 or lo_w > 26:
        return None

    grid_h = random.randint(lo_h, min(26, hi_h))
    grid_w = random.randint(lo_w, min(26, hi_w))

    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(layouts)
    order = sorted(range(len(layouts)), key=lambda i: layouts[i]["total_h"] * layouts[i]["total_w"], reverse=True)

    for idx in order:
        layout = layouts[idx]
        th = int(layout["total_h"])
        tw = int(layout["total_w"])
        if th > grid_h or tw > grid_w:
            return None
        placed = False
        for _ in range(500):
            r0 = random.randint(0, grid_h - th)
            c0 = random.randint(0, grid_w - tw)
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(grid_h, r0 + th + 1)
            cc1 = min(grid_w, c0 + tw + 1)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            occupied[r0 : r0 + th, c0 : c0 + tw] = True
            placements[idx] = (r0, c0)
            placed = True
            break
        if not placed:
            return None

    if any(p is None for p in placements):
        return None
    final_placements = [(int(r), int(c)) for r, c in placements if r is not None and c is not None]

    inp = np.full((grid_h, grid_w), BG, dtype=int)
    expected = np.full((grid_h, grid_w), BG, dtype=int)

    for layout, (pr, pc) in zip(layouts, final_placements):
        input_tl = (int(pr + layout["input_container_tl"][0]), int(pc + layout["input_container_tl"][1]))
        output_tl = (int(pr + layout["output_container_tl"][0]), int(pc + layout["output_container_tl"][1]))
        input_head = [(int(pr + r), int(pc + c)) for r, c in layout["input_head"]]
        dest = [(int(pr + r), int(pc + c)) for r, c in layout["dest"]]
        output_head = [(int(pr + r), int(pc + c)) for r, c in layout["output_head"]]

        _draw_container(inp, input_tl, layout["inner_h"], layout["inner_w"], layout["head_side"], layout["initial_orange"])
        _draw_line(inp, input_head, HEAD_COLOR)
        _draw_line(inp, dest, HEAD_COLOR)

        _draw_line(expected, dest, HEAD_COLOR)
        _draw_container(
            expected,
            output_tl,
            layout["inner_h"],
            layout["inner_w"],
            layout["head_side"],
            layout["initial_orange"] + layout["grey"],
        )
        _draw_line(expected, output_head, HEAD_COLOR)

    try:
        output = main(inp)
    except Exception:
        return None
    if not np.array_equal(output, expected):
        return None

    metrics = _variant_metrics(inp)
    if metrics["overlap"] < 1 or metrics["distance_dominant"] < 1 or metrics["grey_dominant"] < 1:
        return None
    if metrics["same_as_base"] != 0:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 400 attempts")


def _variant_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.array(grid, dtype=int)
    cnt = Counter(map(int, g.flatten()))
    bg = int(max((c for c in cnt.keys() if int(c) != BORDER), key=lambda c: int(cnt[int(c)])))

    metrics = {
        "overlap": 0,
        "distance_dominant": 0,
        "grey_dominant": 0,
        "equal": 0,
        "same_as_base": 0,
    }
    for cont in _find_containers(g):
        box = cont["box"]
        head = _detect_head(g, box, bg)
        if head is None:
            continue
        inner = g[box[0] + 1 : box[2], box[1] + 1 : box[3]]
        grey_count = int(np.count_nonzero(inner == GREY))
        distance = _find_dest_distance(g, box, int(head["side"]), int(head["color"]), bg)
        base_steps = min(int(distance), int(grey_count))
        combo_steps = max(int(distance), int(grey_count))
        if distance == 0:
            metrics["overlap"] += 1
        elif distance > grey_count:
            metrics["distance_dominant"] += 1
        elif grey_count > distance:
            metrics["grey_dominant"] += 1
        else:
            metrics["equal"] += 1
        if base_steps == combo_steps:
            metrics["same_as_base"] += 1
    return metrics
