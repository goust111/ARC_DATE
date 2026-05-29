# Combo variant of dd6b8c4b (index 14)
# combo_spec: D9/1+D15/2
# Name: 多层嵌套框架+BFS逐层收集

import random
from collections import deque

import numpy as np

from arc_utils import neighbors4


# Design analysis for the combo variant
# -------------------------------------
# combo_intent:
#   - Keep the original 11x11 canvas, fixed 3x3 destination block, and the same color roles.
#   - Replace the old loose wall barrier with 2 or 3 concentric rectangular magenta frames.
#   - Replace the old "all reachable maroons are already passable, then sort once by distance" rule
#     with iterative BFS-layer collection:
#       1. Only orange floor is traversable during the search.
#       2. A maroon cell is collected when it sits on the current BFS frontier, i.e. it is 4-neighbor
#          adjacent to the floor region currently reachable from the destination opening.
#       3. Collected maroons turn into floor, so the reachable floor region expands.
#       4. Repeat layer by layer until the 3x3 destination is full or no frontier maroons remain.
#   - This makes maroon cells behave like removable gate tokens instead of already-open corridor cells.
#
# concrete_behavior_trace:
#   - Original task on the same geometry:
#       * Treat floor and maroon as passable from the start.
#       * Flood-fill once through all nested openings.
#       * Sort every reachable maroon by raw BFS distance and move the closest ones.
#   - Combo task:
#       * The innermost frame still has one floor opening next to the destination, so BFS can begin.
#       * Just beyond that opening, one maroon gate blocks access to the first annulus.
#       * After removing that gate, the first annulus becomes reachable; maroons sitting in that annulus
#         and the next frame's gate form the next BFS layer.
#       * Removing that whole layer opens the next annulus, and so on toward the outer region.
#   - Difference:
#       * Reachability is no longer static. A farther maroon that is "close" under the original solver
#         can still wait until an inner gate disappears.
#       * The frames matter jointly with the new BFS rule because every frame opening hosts or leads to a
#         gate maroon that must be peeled away before the next depth becomes reachable.
#
# complexity_calibration:
#   - The grid is too small for many unrelated freedoms, so the generator uses only five frame templates:
#     two 2-layer templates, one 2-layer square template, and two 3-layer zig-zag templates.
#   - Every generated sample contains at least three dynamic collection layers and more than nine reachable
#     maroons, so the solver must reason about both partial final-layer filling and nested unlocking.
#   - The generator prefers far-from-entry maroons in inner annuli and near-entry maroons in outer annuli,
#     then explicitly rejects samples where the combo output matches the original nearest-first output.
#
# main_assumption_audit:
#   - Unique red center and the 3x3 destination block remain unchanged; the original center-finding logic
#     still applies.
#   - The ring around the destination is still dominated by magenta, with exactly one orange opening, so
#     inferring wall/floor from that ring remains valid.
#   - The original assumption "maroon is globally passable" is no longer valid. The combo solver must BFS
#     through floor only, and must mutate a working grid between layers.
#   - Layer ordering must be based on visible structure only. Within each BFS layer the solver uses row-major
#     order, which is human-visible and stable. The generator guarantees that earlier layers exist because:
#       * gate 1 is the only exit from the inner opening,
#       * each outer frame has exactly one gate opening,
#       * no gate overlaps a wall or the destination.
#
# placement_legality:
#   - All frames are axis-aligned rectangles fully inside the 11x11 grid.
#   - Openings are selected from non-corner frame cells only.
#   - Gate maroons are reserved on floor cells known to be inside bounds and not in the destination block.
#   - Source maroons are sampled from disjoint floor-cell candidate sets; distractors, when used, are sampled
#     from the remaining unused floor cells.
#
# visual_observability:
#   - A human sees nested magenta frames with different opening sides, plus maroons clustered by depth.
#   - The output effect is not just "take the closest maroons": inner gate tokens disappear first, then the
#     next annulus empties, then an outer annulus or border strip contributes later tokens.
#   - The generator measures degeneration directly by comparing the combo output against the original solver's
#     nearest-first output on the same input and rejects equal-output samples.


GRID_H, GRID_W = 11, 11
CENTER_R, CENTER_C = 5, 5
FLOOR_COLOR, WALL_COLOR = 7, 6
DEST_COLOR, RED_CENTER = 3, 2
MAROON = 9

DEST_CAPACITY = 9
TOTAL_MAROON_RANGE = (10, 15)
DISTRACTOR_MAX = 3
GEN_RETRIES = 500

FRAME_TEMPLATES = [
    {
        "name": "two_square",
        "boxes": [(3, 3, 7, 7), (1, 1, 9, 9)],
        "weight": 1,
    },
    {
        "name": "two_wide",
        "boxes": [(3, 3, 7, 7), (1, 0, 9, 10)],
        "weight": 2,
    },
    {
        "name": "two_tall",
        "boxes": [(3, 3, 7, 7), (0, 1, 10, 9)],
        "weight": 2,
    },
    {
        "name": "three_wide",
        "boxes": [(3, 3, 7, 7), (3, 1, 7, 9), (1, 1, 9, 9)],
        "weight": 4,
    },
    {
        "name": "three_tall",
        "boxes": [(3, 3, 7, 7), (1, 3, 9, 7), (1, 1, 9, 9)],
        "weight": 4,
    },
]


def _dest_cells() -> list[tuple[int, int]]:
    return [
        (r, c)
        for r in range(CENTER_R - 1, CENTER_R + 2)
        for c in range(CENTER_C - 1, CENTER_C + 2)
    ]


def _frame_cells(box: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _side_positions(box: tuple[int, int, int, int], side: str) -> list[tuple[int, int]]:
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


def _choose_template() -> dict:
    return random.choices(
        FRAME_TEMPLATES,
        weights=[tpl["weight"] for tpl in FRAME_TEMPLATES],
        k=1,
    )[0]


def _choose_openings(template: dict) -> list[tuple[int, int]]:
    name = template["name"]
    boxes = template["boxes"]

    if name == "two_square":
        inner_side = random.choice(["top", "bottom", "left", "right"])
        outer_side = random.choice([side for side in ["top", "bottom", "left", "right"] if side != inner_side])
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


def _depth_outside_boxes(cell: tuple[int, int], boxes: list[tuple[int, int, int, int]]) -> int:
    r, c = cell
    depth = 0
    for r0, c0, r1, c1 in boxes:
        if r < r0 or r > r1 or c < c0 or c > c1:
            depth += 1
    return depth


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _infer_floor_and_wall(grid: np.ndarray, dest_cells: list[tuple[int, int]], dest_set: set[tuple[int, int]]) -> tuple[int, int]:
    ring_colors: list[int] = []
    for r, c in dest_cells:
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1] and (rr, cc) not in dest_set:
                ring_colors.append(int(grid[rr, cc]))

    ring_non_special = [x for x in ring_colors if x not in (RED_CENTER, DEST_COLOR, MAROON)]
    assert ring_non_special, "ring around destination must contain wall/floor colors"
    vals, cnts = np.unique(ring_non_special, return_counts=True)
    wall = int(vals[cnts.argmax()])

    floor_candidates = [int(x) for x in np.unique(grid) if int(x) not in (RED_CENTER, DEST_COLOR, MAROON, wall)]
    if floor_candidates:
        floor = floor_candidates[0]
    else:
        floor = next(int(x) for x in ring_non_special if int(x) != wall)
    return floor, wall


def _floor_starts(grid: np.ndarray, dest_cells: list[tuple[int, int]], floor: int) -> list[tuple[int, int]]:
    h, w = grid.shape
    starts: list[tuple[int, int]] = []
    seen = set()
    for r, c in dest_cells:
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == floor and (rr, cc) not in seen:
                seen.add((rr, cc))
                starts.append((rr, cc))
    return starts


def _reachable_floor(grid: np.ndarray, dest_cells: list[tuple[int, int]], floor: int) -> set[tuple[int, int]]:
    h, w = grid.shape
    starts = _floor_starts(grid, dest_cells, floor)
    assert starts, "there must be at least one floor cell adjacent to the destination block"

    seen = set(starts)
    q = deque(starts)
    while q:
        r, c = q.popleft()
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in seen and int(grid[rr, cc]) == floor:
                seen.add((rr, cc))
                q.append((rr, cc))
    return seen


def _layerwise_sources(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[list[tuple[int, int]]], int]:
    grid = np.asarray(grid, dtype=int)
    centers = np.argwhere(grid == RED_CENTER)
    assert len(centers) == 1, "grid must have exactly one red (2) center pixel"
    cr, cc = map(int, centers[0])
    dest_cells = [(r, c) for r in range(cr - 1, cr + 2) for c in range(cc - 1, cc + 2)]
    dest_cells.sort()
    dest_set = set(dest_cells)
    floor, _wall = _infer_floor_and_wall(grid, dest_cells, dest_set)

    work = grid.copy()
    chosen: list[tuple[int, int]] = []
    layers: list[list[tuple[int, int]]] = []

    while len(chosen) < len(dest_cells):
        reachable = _reachable_floor(work, dest_cells, floor)
        frontier = sorted(
            [
                (r, c)
                for r, c in np.argwhere(work == MAROON)
                if any((nr, nc) in reachable for nr, nc in neighbors4(int(r), int(c)))
            ]
        )
        if not frontier:
            break
        remaining = len(dest_cells) - len(chosen)
        take = frontier[:remaining]
        layers.append(take)
        chosen.extend(take)
        for r, c in take:
            work[r, c] = floor

    return chosen, layers, floor


def _original_nearest_sources(grid: np.ndarray) -> tuple[list[tuple[int, int]], int]:
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    centers = np.argwhere(grid == RED_CENTER)
    assert len(centers) == 1
    cr, cc = map(int, centers[0])

    dest_cells = [(r, c) for r in range(cr - 1, cr + 2) for c in range(cc - 1, cc + 2)]
    dest_cells.sort()
    dest_set = set(dest_cells)
    floor, _wall = _infer_floor_and_wall(grid, dest_cells, dest_set)

    passable = {floor, MAROON}
    starts: list[tuple[int, int]] = []
    seen = set()
    for r, c in dest_cells:
        for rr, cc2 in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc2 < w and int(grid[rr, cc2]) in passable and (rr, cc2) not in seen:
                seen.add((rr, cc2))
                starts.append((rr, cc2))
    assert starts, "there must be at least one passable cell adjacent to the destination block"

    inf = 10**9
    dist = np.full((h, w), inf, dtype=int)
    q = deque(starts)
    for r, c in starts:
        dist[r, c] = 0

    while q:
        r, c = q.popleft()
        d = int(dist[r, c])
        for rr, cc2 in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc2 < w and int(grid[rr, cc2]) in passable and int(dist[rr, cc2]) == inf:
                dist[rr, cc2] = d + 1
                q.append((rr, cc2))

    maroons = [(int(dist[r, c]), int(r), int(c)) for r, c in np.argwhere(grid == MAROON)]
    maroons = [t for t in maroons if t[0] < inf]
    maroons.sort()
    chosen = [(r, c) for _, r, c in maroons[: len(dest_cells)]]
    return chosen, floor


def _build_output_from_sources(grid: np.ndarray, chosen: list[tuple[int, int]], floor: int) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    centers = np.argwhere(out == RED_CENTER)
    assert len(centers) == 1
    cr, cc = map(int, centers[0])
    dest_cells = [(r, c) for r in range(cr - 1, cr + 2) for c in range(cc - 1, cc + 2)]
    dest_cells.sort()

    for r, c in chosen:
        out[r, c] = floor
    for r, c in dest_cells[: len(chosen)]:
        out[r, c] = MAROON
    return out


def _original_style_output(grid: np.ndarray) -> np.ndarray:
    chosen, floor = _original_nearest_sources(grid)
    return _build_output_from_sources(grid, chosen, floor)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pull maroon pixels into the 3x3 block around the red center by unlocking the nested frames layer by layer.

    Input structure:
    - Orange floor surrounds a fixed green 3x3 destination block with one red center.
    - Magenta pixels form 2 or 3 concentric rectangular frames with staggered openings.
    - Maroon pixels sit on floor cells: some act as gate tokens near openings, while others are scattered in annuli or the outer border strip.

    Rule:
    - Start a BFS from floor cells adjacent to the destination, but traverse orange floor only.
    - At each step, collect every maroon that lies on the current BFS frontier, in row-major order; the collected maroons become floor, which may open the next annulus.
    - Continue outward layer by layer until the destination block is full or no frontier maroons remain, then place the collected maroons into the destination in row-major order.

    Key insight: The nested magenta frames do not just filter static reachability. Their openings create a sequence of locked annuli, and maroon gates must be removed in BFS layers before farther maroons can even become candidates.
    """

    chosen, _layers, floor = _layerwise_sources(grid)
    return _build_output_from_sources(grid, chosen, floor)


def _opening_outward_neighbor(box: tuple[int, int, int, int], opening: tuple[int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    r, c = opening
    if r == r0:
        return r - 1, c
    if r == r1:
        return r + 1, c
    if c == c0:
        return r, c - 1
    if c == c1:
        return r, c + 1
    raise ValueError("opening is not on the frame boundary")


def _layout_depth_groups(grid: np.ndarray, boxes: list[tuple[int, int, int, int]], dest_set: set[tuple[int, int]]) -> dict[int, list[tuple[int, int]]]:
    groups: dict[int, list[tuple[int, int]]] = {d: [] for d in range(1, len(boxes) + 1)}
    for r in range(GRID_H):
        for c in range(GRID_W):
            cell = (r, c)
            if cell in dest_set or int(grid[r, c]) != FLOOR_COLOR:
                continue
            depth = _depth_outside_boxes(cell, boxes)
            if depth in groups:
                groups[depth].append(cell)
    for depth in groups:
        groups[depth].sort()
    return groups


def _build_layout() -> dict | None:
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
    wall_cells -= opening_set
    for r, c in wall_cells:
        grid[r, c] = WALL_COLOR

    gate_cells = []
    first_gate = _opening_outward_neighbor(boxes[0], openings[0])
    if not (0 <= first_gate[0] < GRID_H and 0 <= first_gate[1] < GRID_W):
        return None
    gate_cells.append(first_gate)
    gate_cells.extend(openings[1:])
    if any(cell in dest_set or int(grid[cell]) != FLOOR_COLOR for cell in gate_cells):
        return None

    depth_groups = _layout_depth_groups(grid, boxes, dest_set)
    layer_candidates: dict[int, list[tuple[int, int]]] = {}
    reserved = set(gate_cells) | opening_set
    for depth in range(1, len(boxes) + 1):
        layer_candidates[depth] = [cell for cell in depth_groups[depth] if cell not in reserved]

    if any(len(depth_groups[depth]) == 0 for depth in depth_groups):
        return None
    if len(layer_candidates[len(boxes)]) < 3:
        return None

    return {
        "grid": grid,
        "boxes": boxes,
        "openings": openings,
        "gate_cells": gate_cells,
        "layer_candidates": layer_candidates,
        "template_name": template["name"],
        "dest_cells": dest_cells,
        "dest_set": dest_set,
    }


def _pick_layer_tokens(layout: dict) -> list[tuple[int, int]] | None:
    boxes = layout["boxes"]
    openings = layout["openings"]
    gates = layout["gate_cells"]
    candidates = layout["layer_candidates"]

    chosen_layers: list[list[tuple[int, int]]] = [[gates[0]]]
    used = {gates[0]}

    for depth in range(1, len(boxes)):
        gate = gates[depth]
        pool = [cell for cell in candidates[depth] if cell not in used]
        if len(pool) < 2:
            return None
        entry = gates[depth - 1]
        pool.sort(key=lambda cell: (_manhattan(cell, entry), cell[0], cell[1]), reverse=True)
        extra_count = random.randint(2, min(4, len(pool)))
        picked = sorted(random.sample(pool[: max(extra_count + 1, min(len(pool), 8))], extra_count))
        layer = sorted(picked + [gate])
        chosen_layers.append(layer)
        used.update(layer)

    outer_depth = len(boxes)
    outer_pool = [cell for cell in candidates[outer_depth] if cell not in used]
    if len(outer_pool) < 4:
        return None
    entry = openings[-1]
    outer_pool.sort(key=lambda cell: (_manhattan(cell, entry), cell[0], cell[1]))
    outer_count = random.randint(4, min(6, len(outer_pool)))
    outer_layer = sorted(random.sample(outer_pool[: max(outer_count + 1, min(len(outer_pool), 10))], outer_count))
    chosen_layers.append(outer_layer)

    flat = [cell for layer in chosen_layers for cell in layer]
    if len(flat) < TOTAL_MAROON_RANGE[0]:
        return None
    if len(flat) > TOTAL_MAROON_RANGE[1]:
        return None
    if len(chosen_layers) < 3:
        return None
    return flat


def _pick_distractors(layout: dict, used: set[tuple[int, int]]) -> list[tuple[int, int]]:
    grid = layout["grid"]
    dest_set = layout["dest_set"]
    boxes = layout["boxes"]
    candidates = []
    for r in range(GRID_H):
        for c in range(GRID_W):
            cell = (r, c)
            if cell in used or cell in dest_set or int(grid[r, c]) != FLOOR_COLOR:
                continue
            if _depth_outside_boxes(cell, boxes) == len(boxes):
                candidates.append(cell)
    if not candidates:
        return []
    count = random.randint(0, min(DISTRACTOR_MAX, len(candidates)))
    return random.sample(candidates, count)


def _fusion_metrics(inp: np.ndarray) -> dict:
    chosen, layers, floor = _layerwise_sources(inp)
    original_sources, _ = _original_nearest_sources(inp)
    original = _original_style_output(inp)
    combo = _build_output_from_sources(inp, chosen, floor)
    source_symdiff = len(set(chosen) ^ set(original_sources))
    return {
        "layer_count": len(layers),
        "moved_count": len(chosen),
        "layer_sizes": [len(layer) for layer in layers],
        "source_symdiff": source_symdiff,
        "differs_from_original": int(not np.array_equal(combo, original)),
    }


def _gen_attempt() -> np.ndarray | None:
    layout = _build_layout()
    if layout is None:
        return None

    source_cells = _pick_layer_tokens(layout)
    if source_cells is None:
        return None

    inp = layout["grid"].copy()
    used = set(source_cells)
    distractors = _pick_distractors(layout, used)
    for r, c in source_cells + distractors:
        inp[r, c] = MAROON

    chosen, layers, floor = _layerwise_sources(inp)
    if len(chosen) != DEST_CAPACITY:
        return None
    if len(layers) < 3:
        return None

    combo = _build_output_from_sources(inp, chosen, floor)
    original = _original_style_output(inp)
    if np.array_equal(combo, original):
        return None
    if not np.array_equal(main(inp), combo):
        return None

    original_sources, _ = _original_nearest_sources(inp)
    if set(original_sources) == set(chosen):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
