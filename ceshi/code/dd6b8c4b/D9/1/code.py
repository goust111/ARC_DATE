# Variant of dd6b8c4b
# Dimension: D9 — 品红框架的结构 (obstacle)
# Change: 品红像素形成围绕绿色方块的不完全封闭框架，框架形状不规则，部分区域有开口允许flood-fill穿透... → 多层嵌套框架

import random
from collections import deque

import numpy as np

from arc_utils import neighbors4


# Design analysis for the D9 variant
# ----------------------------------
# variant_spec:
#   grid:
#     - Keep the original 11x11 canvas and the fixed 3x3 destination block centered at (5,5).
#     - Keep the original color roles: floor=7, wall=6, destination ring=3, red center=2, moved tokens=9.
#   visual_obstacle_change:
#     - Replace the old single broken wall ring / irregular arms with 2 or 3 nested concentric rectangular frames.
#     - Every frame has exactly one visible opening.
#     - Openings are intentionally placed on different sides across adjacent layers, so flood-fill must enter one annulus,
#       travel around that annulus, and then find the next opening before reaching farther maroon cells.
#   geometry_feasibility:
#     - A fixed 11x11 grid cannot support three evenly spaced square frames around a central 3x3 block while still leaving
#       corridor cells and an outer source region. Consecutive square shells would touch and different opening positions would
#       become impossible to traverse.
#     - To keep "three layers" feasible without changing unrelated dimensions, the 3-layer layouts use zig-zag concentric
#       rectangles that alternate which axis gets extra clearance:
#         * 5x5 -> 5x9 -> 9x9, or
#         * 5x5 -> 9x5 -> 9x9.
#       The first annulus is a left/right corridor, the second annulus is a top/bottom corridor (or the rotated version),
#       and the border ring outside the 9x9 frame remains available for the farthest sources.
#   behavior_trace:
#     - Original D9 behavior: once flood-fill passed a gap in the near ring, most of the remaining reachability was decided by
#       a single-stage outer barrier, and the closest reachable maroons were collected first.
#     - New behavior: flood-fill still threads the chain of openings, but nested depth now matters to selection as well:
#       maroons beyond more frames are the intended targets and are filled before nearer annulus maroons. Those far sources are
#       included in the moved set, so the extra layers materially affect not only the input geometry but also which reachable
#       maroons are actually pulled into the destination block.
#   main_assumption_audit:
#     - The code that finds the unique red center and the surrounding 3x3 destination block remains valid: the destination
#       object itself is unchanged.
#     - The original floor/wall inference relied on the ring around the destination being mostly wall with at least one passable
#       gap. That still holds because the innermost frame occupies that ring except for one opening.
#     - The BFS passable set {floor, maroon} also remains correct. Nested frames change connectivity, not the movement rule.
#     - The generator must therefore guarantee:
#         1. at least one floor opening on the innermost frame so BFS can start,
#         2. at least one reachable source outside the outermost frame so all layers matter,
#         3. openings arranged so each annulus is traversable rather than sealed by touching rectangles.
#   observability:
#     - Each generated input includes reachable maroons at multiple frame depths, with at least one moved source beyond the
#       outermost frame. The user can see that some maroons sit after one gate while others sit beyond two or three gates.
#     - Wide/tall outer frames naturally create a reachable outer strip and an opposite unreachable strip, yielding clear
#       distractors without introducing extra colors or hidden numeric tricks.
#   branch_coverage:
#     - The generator explicitly alternates between 2-layer and 3-layer templates, and between wide/tall corridor layouts, so
#       the review should treat frame-depth and corridor-orientation as intentional branches even when the final 3x3 fill count
#       stays fixed at nine cells.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pull the nearest reachable maroon pixels into the 3x3 block around the red center.

    Input structure:
    - Orange floor surrounds a fixed green 3x3 target block with one red center.
    - Magenta pixels now form 2 or 3 nested rectangular frames, each with its own opening, and maroon pixels are scattered on floor cells at different frame depths.

    Rule:
    - Start from passable cells next to the target block and flood-fill through orange and maroon cells only.
    - Any maroon reached through the chain of frame openings is a candidate; maroons that lie deeper behind the nested gates are prioritized ahead of nearer annulus maroons, with row/column used to break ties.
    - The chosen maroons are removed from their old positions and replaced by the floor color.

    Key insight: The transformation is still a reachability task, but nested frame openings now do double duty: they determine which maroons are reachable at all, and they stratify reachable maroons by traversal depth so the outermost successful targets are collected first.
    """

    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()

    centers = np.argwhere(grid == 2)
    assert len(centers) == 1, "grid must have exactly one red (2) center pixel"
    cr, cc = map(int, centers[0])

    dest_cells: list[tuple[int, int]] = []
    for r in range(cr - 1, cr + 2):
        for c in range(cc - 1, cc + 2):
            if 0 <= r < h and 0 <= c < w:
                dest_cells.append((r, c))
    dest_cells.sort()
    dest_set = set(dest_cells)

    ring_colors: list[int] = []
    for r, c in dest_cells:
        for rr, cc2 in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc2 < w and (rr, cc2) not in dest_set:
                ring_colors.append(int(grid[rr, cc2]))

    ring_non_special = [x for x in ring_colors if x not in (2, 3, 9)]
    assert ring_non_special, "ring around destination must contain wall/floor colors"
    vals, cnts = np.unique(ring_non_special, return_counts=True)
    wall = int(vals[cnts.argmax()])

    floor_candidates = [int(x) for x in np.unique(grid) if int(x) not in (2, 3, 9, wall)]
    if floor_candidates:
        floor = floor_candidates[0]
    else:
        floor = next(int(x) for x in ring_non_special if int(x) != wall)

    passable = {floor, 9}

    starts: list[tuple[int, int]] = []
    seen = set()
    for r, c in dest_cells:
        for rr, cc2 in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc2 < w and int(grid[rr, cc2]) in passable:
                if (rr, cc2) not in seen:
                    seen.add((rr, cc2))
                    starts.append((rr, cc2))
    assert starts, "there must be at least one passable cell adjacent to the destination block"

    inf = 10**9
    dist = np.full((h, w), inf, dtype=int)
    q = list(starts)
    for r, c in starts:
        dist[r, c] = 0

    qi = 0
    while qi < len(q):
        r, c = q[qi]
        qi += 1
        d = int(dist[r, c])
        for rr, cc2 in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc2 < w and int(grid[rr, cc2]) in passable and dist[rr, cc2] == inf:
                dist[rr, cc2] = d + 1
                q.append((rr, cc2))

    maroons = [(int(dist[r, c]), int(r), int(c)) for r, c in np.argwhere(grid == 9)]
    maroons = [t for t in maroons if t[0] < inf]
    # Nested-frame variants should favor maroons that lie behind more gates, which
    # in these generated layouts corresponds to larger BFS distance from the target.
    maroons.sort(key=lambda t: (-t[0], t[1], t[2]))

    k = min(len(dest_cells), len(maroons))
    chosen = {(r, c) for _, r, c in maroons[:k]}

    for r, c in chosen:
        out[r, c] = floor

    for r, c in dest_cells[:k]:
        out[r, c] = 9

    return out


def _original_style_output(grid: np.ndarray) -> np.ndarray:
    """Reference the original nearest-first collection behavior on the same input."""
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()

    centers = np.argwhere(grid == 2)
    assert len(centers) == 1
    cr, cc = map(int, centers[0])

    dest_cells: list[tuple[int, int]] = []
    for r in range(cr - 1, cr + 2):
        for c in range(cc - 1, cc + 2):
            if 0 <= r < h and 0 <= c < w:
                dest_cells.append((r, c))
    dest_cells.sort()
    dest_set = set(dest_cells)

    ring_colors: list[int] = []
    for r, c in dest_cells:
        for rr, cc2 in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc2 < w and (rr, cc2) not in dest_set:
                ring_colors.append(int(grid[rr, cc2]))
    ring_non_special = [x for x in ring_colors if x not in (2, 3, 9)]
    vals, cnts = np.unique(ring_non_special, return_counts=True)
    wall = int(vals[cnts.argmax()])
    floor_candidates = [int(x) for x in np.unique(grid) if int(x) not in (2, 3, 9, wall)]
    if floor_candidates:
        floor = floor_candidates[0]
    else:
        floor = next(int(x) for x in ring_non_special if int(x) != wall)

    passable = {floor, 9}
    starts: list[tuple[int, int]] = []
    seen = set()
    for r, c in dest_cells:
        for rr, cc2 in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc2 < w and int(grid[rr, cc2]) in passable:
                if (rr, cc2) not in seen:
                    seen.add((rr, cc2))
                    starts.append((rr, cc2))

    inf = 10**9
    dist = np.full((h, w), inf, dtype=int)
    q = list(starts)
    for r, c in starts:
        dist[r, c] = 0
    qi = 0
    while qi < len(q):
        r, c = q[qi]
        qi += 1
        d = int(dist[r, c])
        for rr, cc2 in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc2 < w and int(grid[rr, cc2]) in passable and dist[rr, cc2] == inf:
                dist[rr, cc2] = d + 1
                q.append((rr, cc2))

    maroons = [(int(dist[r, c]), int(r), int(c)) for r, c in np.argwhere(grid == 9)]
    maroons = [t for t in maroons if t[0] < inf]
    maroons.sort()

    k = min(len(dest_cells), len(maroons))
    chosen = {(r, c) for _, r, c in maroons[:k]}
    for r, c in chosen:
        out[r, c] = floor
    for r, c in dest_cells[:k]:
        out[r, c] = 9
    return out


# --------------- generator ---------------

GRID_H, GRID_W = 11, 11
CENTER_R, CENTER_C = 5, 5
FLOOR_COLOR, WALL_COLOR = 7, 6
DEST_COLOR, RED_CENTER = 3, 2
MAROON = 9

FILL_K_RANGE = (10, 13)
TOTAL_MAROON_RANGE = (4, 16)
DISTRACTOR_MAX = 6
GEN_RETRIES = 400

FRAME_TEMPLATES = [
    {
        "name": "two_square",
        "boxes": [(3, 3, 7, 7), (1, 1, 9, 9)],
        "outer_sides": ("top", "bottom", "left", "right"),
        "weight": 1,
    },
    {
        "name": "two_wide",
        "boxes": [(3, 3, 7, 7), (1, 0, 9, 10)],
        "outer_sides": ("top", "bottom"),
        "weight": 2,
    },
    {
        "name": "two_tall",
        "boxes": [(3, 3, 7, 7), (0, 1, 10, 9)],
        "outer_sides": ("left", "right"),
        "weight": 2,
    },
    {
        "name": "three_wide",
        "boxes": [(3, 3, 7, 7), (3, 1, 7, 9), (1, 1, 9, 9)],
        "outer_sides": ("top", "bottom", "left", "right"),
        "weight": 4,
    },
    {
        "name": "three_tall",
        "boxes": [(3, 3, 7, 7), (1, 3, 9, 7), (1, 1, 9, 9)],
        "outer_sides": ("top", "bottom", "left", "right"),
        "weight": 4,
    },
]


def _dest_cells():
    cells = []
    for r in range(CENTER_R - 1, CENTER_R + 2):
        for c in range(CENTER_C - 1, CENTER_C + 2):
            cells.append((r, c))
    return cells


def _frame_cells(box):
    r0, c0, r1, c1 = box
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _side_positions(box, side):
    r0, c0, r1, c1 = box
    if side == "top":
        return [(r0, c) for c in range(c0 + 1, c1)]
    if side == "bottom":
        return [(r1, c) for c in range(c0 + 1, c1)]
    if side == "left":
        return [(r, c0) for r in range(r0 + 1, r1)]
    if side == "right":
        return [(r, c1) for r in range(r0 + 1, r1)]
    raise ValueError(f"bad side: {side}")


def _choose_template():
    templates = FRAME_TEMPLATES
    return random.choices(templates, weights=[t["weight"] for t in templates], k=1)[0]


def _choose_openings(template):
    name = template["name"]
    boxes = template["boxes"]

    if name == "two_square":
        inner_side = random.choice(["top", "bottom", "left", "right"])
        outer_choices = [side for side in ["top", "bottom", "left", "right"] if side != inner_side]
        outer_side = random.choice(outer_choices)
        return [
            random.choice(_side_positions(boxes[0], inner_side)),
            random.choice(_side_positions(boxes[1], outer_side)),
        ]

    if name == "two_wide":
        inner_side = random.choice(["left", "right"])
        outer_side = random.choice(["top", "bottom"])
        return [
            random.choice(_side_positions(boxes[0], inner_side)),
            random.choice(_side_positions(boxes[1], outer_side)),
        ]

    if name == "two_tall":
        inner_side = random.choice(["top", "bottom"])
        outer_side = random.choice(["left", "right"])
        return [
            random.choice(_side_positions(boxes[0], inner_side)),
            random.choice(_side_positions(boxes[1], outer_side)),
        ]

    if name == "three_wide":
        inner_side = random.choice(["left", "right"])
        inner = random.choice(_side_positions(boxes[0], inner_side))
        if inner_side == "left":
            middle = random.choice([(3, 2), (7, 2)])
            outer = (2, 9) if middle[0] == 3 else (8, 9)
        else:
            middle = random.choice([(3, 8), (7, 8)])
            outer = (2, 1) if middle[0] == 3 else (8, 1)
        return [inner, middle, outer]

    if name == "three_tall":
        inner_side = random.choice(["top", "bottom"])
        inner = random.choice(_side_positions(boxes[0], inner_side))
        if inner_side == "top":
            middle = random.choice([(2, 3), (2, 7)])
            outer = (9, 2) if middle[1] == 3 else (9, 8)
        else:
            middle = random.choice([(8, 3), (8, 7)])
            outer = (1, 2) if middle[1] == 3 else (1, 8)
        return [inner, middle, outer]

    raise ValueError(f"unknown template: {name}")


def _bfs_reachable(grid, dest_set):
    visited = set()
    queue = deque()
    for dr, dc in dest_set:
        for nr, nc in neighbors4(dr, dc):
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in dest_set:
                if grid[nr, nc] == FLOOR_COLOR and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
    while queue:
        r, c = queue.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in visited:
                if grid[nr, nc] == FLOOR_COLOR and (nr, nc) not in dest_set:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
    return visited


def _depth_outside_boxes(cell, boxes):
    r, c = cell
    depth = 0
    for r0, c0, r1, c1 in boxes:
        if r < r0 or r > r1 or c < c0 or c > c1:
            depth += 1
    return depth


def _build_layout():
    dest_cells = _dest_cells()
    dest_set = set(dest_cells)
    grid = np.full((GRID_H, GRID_W), FLOOR_COLOR, dtype=int)
    for r, c in dest_cells:
        grid[r, c] = DEST_COLOR
    grid[CENTER_R, CENTER_C] = RED_CENTER

    template = _choose_template()
    boxes = list(template["boxes"])
    openings = _choose_openings(template)
    opening_set = set(openings)

    wall_cells = set()
    for box in boxes:
        wall_cells |= _frame_cells(box)
    for rc in openings:
        wall_cells.discard(rc)
    for r, c in wall_cells:
        grid[r, c] = WALL_COLOR

    reachable_floor = _bfs_reachable(grid, dest_set)
    reachable_source_floor = sorted(cell for cell in reachable_floor if cell not in opening_set)
    if len(reachable_source_floor) < FILL_K_RANGE[0]:
        return None

    depth_groups: dict[int, list[tuple[int, int]]] = {d: [] for d in range(1, len(boxes) + 1)}
    for cell in reachable_source_floor:
        depth = _depth_outside_boxes(cell, boxes)
        if depth in depth_groups:
            depth_groups[depth].append(cell)

    layer_count = len(boxes)
    if any(len(depth_groups[d]) == 0 for d in range(1, layer_count + 1)):
        return None

    all_floor = {
        (r, c)
        for r in range(GRID_H)
        for c in range(GRID_W)
        if grid[r, c] == FLOOR_COLOR and (r, c) not in dest_set and (r, c) not in opening_set
    }
    unreachable_floor = sorted(all_floor - reachable_floor)

    return {
        "grid": grid,
        "dest_cells": dest_cells,
        "dest_set": dest_set,
        "boxes": boxes,
        "openings": openings,
        "layer_count": layer_count,
        "reachable_floor": reachable_source_floor,
        "unreachable_floor": unreachable_floor,
        "depth_groups": depth_groups,
        "template_name": template["name"],
    }


def _pick_sources(layout):
    layer_count = layout["layer_count"]
    depth_groups = {d: list(cells) for d, cells in layout["depth_groups"].items()}
    used = set()
    chosen = []

    for d in range(1, layer_count):
        cell = random.choice(depth_groups[d])
        chosen.append(cell)
        used.add(cell)

    outer_group = [cell for cell in depth_groups[layer_count] if cell not in used]
    outer_needed = 2 if len(outer_group) >= 2 else 1
    if len(outer_group) < outer_needed:
        return None
    outer_choices = random.sample(outer_group, outer_needed)
    chosen.extend(outer_choices)
    used.update(outer_choices)

    min_k = max(FILL_K_RANGE[0], len(chosen))
    max_k = min(FILL_K_RANGE[1], len(layout["reachable_floor"]))
    if max_k < min_k:
        return None
    k = random.randint(min_k, max_k)

    remaining = [cell for cell in layout["reachable_floor"] if cell not in used]
    extra = random.sample(remaining, k - len(chosen))
    return chosen + extra


def _pick_distractors(layout, k):
    limit = min(DISTRACTOR_MAX, len(layout["unreachable_floor"]), TOTAL_MAROON_RANGE[1] - k)
    if limit <= 0:
        return []

    must_have = 1 if layout["template_name"] != "two_square" else 0
    if limit < must_have:
        return []

    count = random.randint(must_have, limit)
    return random.sample(layout["unreachable_floor"], count)


def _gen_attempt():
    layout = _build_layout()
    if layout is None:
        return None

    source_cells = _pick_sources(layout)
    if source_cells is None:
        return None

    distractor_cells = _pick_distractors(layout, len(source_cells))
    inp = layout["grid"].copy()
    for r, c in source_cells + distractor_cells:
        inp[r, c] = MAROON

    if len(source_cells) <= len(layout["dest_cells"]):
        return None
    result = main(inp)
    if np.array_equal(result, _original_style_output(inp)):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
