# Combo variant of dd6b8c4b (index 53)
# combo_spec: D2/2+D9/1+D15/2
# Name: 动态墙壁+嵌套框架+BFS逐层收集

from __future__ import annotations

import os
import random
import sys
from collections import deque

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SOLVER_DIR = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "..", "solvers"))
if _SOLVER_DIR not in sys.path:
    sys.path.append(_SOLVER_DIR)

from arc_utils import neighbors4


# Design analysis for the combo variant
# -------------------------------------
# concrete_behavior:
#   - Keep the original 11x11 canvas, the fixed 3x3 green block, and the unique red center at (5,5).
#   - Replace the original loose wall ring with exactly three nested magenta frames, using the feasible
#     3-layer layouts from the D9 reference:
#       * "wide": 5x5 -> 5x9 -> 9x9, so the first accessible annulus is left/right and the second is top/bottom.
#       * "tall": 5x5 -> 9x5 -> 9x9, so the first accessible annulus is top/bottom and the second is left/right.
#   - The innermost frame has one permanent opening. The middle and outer frames start closed.
#   - One maroon trigger sits in the first annulus and one in the second annulus. Collecting the first trigger
#     removes a short gate segment from the middle frame; collecting the second trigger removes a short gate
#     segment from the outer frame. Additional maroons wait outside the outer frame and are only collectible
#     after the second breach.
# behavior_trace:
#   - Original task: run one static BFS, keep all reachable maroons, sort once, then fill the destination.
#   - Combo task:
#       1. BFS sees only the first annulus, so only the inner trigger is reachable.
#       2. Collect the nearest BFS layer containing that trigger; this opens the middle gate.
#       3. Re-run BFS on the updated grid. The second-annulus trigger becomes the next reachable layer.
#       4. Collect that layer; this opens the outer gate.
#       5. Re-run BFS again. The outer-border maroons are now reachable and are collected layer by layer.
#     The final output therefore differs from the static version in two visible ways:
#       * more maroons move into the 3x3 block;
#       * two magenta wall segments have disappeared.
# complexity_calibration:
#   - The variant must look different from the original at a glance, so the generator always uses 3 layers,
#     not 0-1 optional walls.
#   - To avoid ambiguity, the route freedom is capped to 8 templates: 2 orientations x 2 entry sides x 2 turn
#     choices. Each route repeats enough structure that a human can compare examples and notice the same
#     "break shell, then break shell again" pattern.
#   - The first two annuli each contain exactly one trigger maroon. This prevents the dynamic process from
#     degenerating into many competing local breaches that would be hard to infer from 2-4 examples.
# implicit_assumption_audit:
#   - The original red-center detection still holds because the destination object is unchanged.
#   - The original color inference from the ring around the destination still holds because the innermost frame
#     occupies that ring except for one floor opening.
#   - The original assumption "one BFS is enough" is no longer valid, so the solver must iterate:
#       * compute reachable maroons,
#       * take the nearest BFS layer only,
#       * mutate walls,
#       * recompute reachability.
#   - New solver assumptions introduced here:
#       * exactly one route-defining inner opening exists on the innermost frame;
#       * exactly one first-annulus trigger exists among the route candidates;
#       * total collectible maroons never exceeds the 9 destination cells.
#     The generator enforces all three conditions directly.
# placement_legality_and_observability:
#   - All wall boxes are fixed legal rectangles inside the 11x11 grid, so no overlaps or out-of-bounds cases
#     are possible except the intentional frame overlaps required by the template.
#   - Triggers are placed on corridor cells, never on wall or destination cells.
#   - Unreachable distractors are placed in the opposite sealed corridors. They stay visibly maroon in the
#     output, giving the human solver evidence that only specific shell breaks happen.
# visual_cleanliness:
#   - Only five semantic regions are used: destination, walls, first annulus, second annulus, and outside border.
#   - No hidden color arithmetic is used. Every decision is recoverable from visible geometry:
#       * entry side comes from the single opening in the inner frame;
#       * turn direction comes from which first-annulus trigger is present;
#       * later collected maroons line up beyond the newly opened gates.


GRID_H, GRID_W = 11, 11
CENTER_R, CENTER_C = 5, 5
FLOOR_COLOR, WALL_COLOR = 7, 6
DEST_COLOR, RED_CENTER = 3, 2
MAROON = 9

GEN_RETRIES = 400
OUTER_COUNT_RANGE = (4, 7)
DISTRACTOR_RANGE = (1, 3)

BOX1 = (3, 3, 7, 7)
BOX2_WIDE = (3, 1, 7, 9)
BOX2_TALL = (1, 3, 9, 7)
BOX3 = (1, 1, 9, 9)


def _dest_cells() -> list[tuple[int, int]]:
    return [
        (r, c)
        for r in range(CENTER_R - 1, CENTER_R + 2)
        for c in range(CENTER_C - 1, CENTER_C + 2)
    ]


DEST_CELLS = _dest_cells()
DEST_SET = set(DEST_CELLS)


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


BOX1_FRAME = _frame_cells(BOX1)
BOX2_WIDE_FRAME = _frame_cells(BOX2_WIDE)
BOX2_TALL_FRAME = _frame_cells(BOX2_TALL)
BOX3_FRAME = _frame_cells(BOX3)


def _inside(box: tuple[int, int, int, int], cell: tuple[int, int]) -> bool:
    r, c = cell
    r0, c0, r1, c1 = box
    return r0 <= r <= r1 and c0 <= c <= c1


def _outside_border_cells() -> list[tuple[int, int]]:
    cells = []
    for r in range(GRID_H):
        for c in range(GRID_W):
            if not _inside(BOX3, (r, c)) and (r, c) not in DEST_SET:
                cells.append((r, c))
    return sorted(cells)


OUTSIDE_CELLS = _outside_border_cells()


def _build_routes() -> list[dict]:
    routes = []

    def add_route(name, orientation, opening, shell1, gate2, shell2, gate3, distractors):
        routes.append(
            {
                "name": name,
                "orientation": orientation,
                "boxes": (BOX1, BOX2_WIDE if orientation == "wide" else BOX2_TALL, BOX3),
                "opening": opening,
                "shell1": shell1,
                "gate2": tuple(gate2),
                "shell2": shell2,
                "gate3": tuple(gate3),
                "distractors": tuple(sorted(distractors)),
                "outer_pool": tuple(OUTSIDE_CELLS),
            }
        )

    add_route(
        "wide_left_top",
        "wide",
        (5, 3),
        (4, 2),
        [(3, 1), (3, 2)],
        (2, 8),
        [(1, 9), (2, 9)],
        {(4, 8), (5, 8), (6, 8), (8, 2), (8, 4), (8, 6), (8, 8)},
    )
    add_route(
        "wide_left_bottom",
        "wide",
        (5, 3),
        (6, 2),
        [(7, 1), (7, 2)],
        (8, 8),
        [(8, 9), (9, 9)],
        {(4, 8), (5, 8), (6, 8), (2, 2), (2, 4), (2, 6), (2, 8)},
    )
    add_route(
        "wide_right_top",
        "wide",
        (5, 7),
        (4, 8),
        [(3, 8), (3, 9)],
        (2, 2),
        [(1, 1), (2, 1)],
        {(4, 2), (5, 2), (6, 2), (8, 2), (8, 4), (8, 6), (8, 8)},
    )
    add_route(
        "wide_right_bottom",
        "wide",
        (5, 7),
        (6, 8),
        [(7, 8), (7, 9)],
        (8, 2),
        [(8, 1), (9, 1)],
        {(4, 2), (5, 2), (6, 2), (2, 2), (2, 4), (2, 6), (2, 8)},
    )
    add_route(
        "tall_top_left",
        "tall",
        (3, 5),
        (2, 4),
        [(1, 3), (2, 3)],
        (8, 2),
        [(9, 1), (9, 2)],
        {(8, 4), (8, 5), (8, 6), (2, 8), (4, 8), (6, 8), (8, 8)},
    )
    add_route(
        "tall_top_right",
        "tall",
        (3, 5),
        (2, 6),
        [(1, 7), (2, 7)],
        (8, 8),
        [(9, 8), (9, 9)],
        {(8, 4), (8, 5), (8, 6), (2, 2), (4, 2), (6, 2), (8, 2)},
    )
    add_route(
        "tall_bottom_left",
        "tall",
        (7, 5),
        (8, 4),
        [(8, 3), (9, 3)],
        (2, 2),
        [(1, 1), (1, 2)],
        {(2, 4), (2, 5), (2, 6), (2, 8), (4, 8), (6, 8), (8, 8)},
    )
    add_route(
        "tall_bottom_right",
        "tall",
        (7, 5),
        (8, 6),
        [(8, 7), (9, 7)],
        (2, 8),
        [(1, 8), (1, 9)],
        {(2, 4), (2, 5), (2, 6), (2, 2), (4, 2), (6, 2), (8, 2)},
    )

    return routes


ROUTES = _build_routes()


def _infer_floor_and_wall(grid: np.ndarray, dest_cells: list[tuple[int, int]], dest_set: set[tuple[int, int]]) -> tuple[int, int]:
    ring_colors = []
    for r, c in dest_cells:
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < GRID_H and 0 <= cc < GRID_W and (rr, cc) not in dest_set:
                ring_colors.append(int(grid[rr, cc]))

    ring_non_special = [x for x in ring_colors if x not in (RED_CENTER, DEST_COLOR, MAROON)]
    assert ring_non_special, "ring around destination must contain both floor and wall colors"
    vals, cnts = np.unique(ring_non_special, return_counts=True)
    wall = int(vals[cnts.argmax()])
    floor_candidates = [int(x) for x in np.unique(grid) if int(x) not in (RED_CENTER, DEST_COLOR, MAROON, wall)]
    if floor_candidates:
        floor = floor_candidates[0]
    else:
        floor = next(int(x) for x in ring_non_special if int(x) != wall)
    return floor, wall


def _bfs_dist(grid: np.ndarray, dest_cells: list[tuple[int, int]], dest_set: set[tuple[int, int]], passable: set[int]) -> np.ndarray:
    inf = 10**9
    dist = np.full(grid.shape, inf, dtype=int)
    q: deque[tuple[int, int]] = deque()
    seen = set()
    for r, c in dest_cells:
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < GRID_H and 0 <= cc < GRID_W and int(grid[rr, cc]) in passable:
                if (rr, cc) not in seen:
                    seen.add((rr, cc))
                    dist[rr, cc] = 0
                    q.append((rr, cc))

    assert q, "there must be at least one passable entry next to the destination block"

    while q:
        r, c = q.popleft()
        d = int(dist[r, c])
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < GRID_H and 0 <= cc < GRID_W and int(grid[rr, cc]) in passable and dist[rr, cc] == inf:
                dist[rr, cc] = d + 1
                q.append((rr, cc))
    return dist


def _zone_of(cell: tuple[int, int], route: dict) -> int:
    _box1, box2, box3 = route["boxes"]
    if _inside(box2, cell) and not _inside(BOX1, cell):
        return 1
    if _inside(box3, cell) and not _inside(box2, cell):
        return 2
    if not _inside(box3, cell):
        return 3
    return 0


def _detect_route(grid: np.ndarray, wall: int) -> dict:
    wide_markers = {(3, 2), (3, 8), (7, 2), (7, 8)}
    tall_markers = {(2, 3), (2, 7), (8, 3), (8, 7)}
    wide_score = sum(int(grid[r, c] == wall) for r, c in wide_markers)
    tall_score = sum(int(grid[r, c] == wall) for r, c in tall_markers)
    orientation = "wide" if wide_score > tall_score else "tall"

    candidates = [route for route in ROUTES if route["orientation"] == orientation]
    matches = []
    for route in candidates:
        opening_r, opening_c = route["opening"]
        shell1_r, shell1_c = route["shell1"]
        if int(grid[opening_r, opening_c]) == wall:
            continue
        if int(grid[shell1_r, shell1_c]) != MAROON:
            continue
        matches.append(route)

    assert len(matches) == 1, "input must encode exactly one combo route"
    return matches[0]


def _simulate(grid: np.ndarray) -> dict:
    grid = np.asarray(grid, dtype=int)
    centers = np.argwhere(grid == RED_CENTER)
    assert len(centers) == 1, "grid must have exactly one red center"
    floor, wall = _infer_floor_and_wall(grid, DEST_CELLS, DEST_SET)
    route = _detect_route(grid, wall)
    passable = {floor, MAROON}

    work = grid.copy()
    collected: list[tuple[int, int]] = []
    opened_segments: list[tuple[tuple[int, int], ...]] = []
    stage = 0

    first_dist = _bfs_dist(work, DEST_CELLS, DEST_SET, passable)
    initial_reachable = sorted(
        (int(r), int(c))
        for r, c in np.argwhere(work == MAROON)
        if int(first_dist[r, c]) < 10**9
    )

    while len(collected) < len(DEST_CELLS):
        dist = _bfs_dist(work, DEST_CELLS, DEST_SET, passable)
        reachable = [
            (int(dist[r, c]), int(r), int(c))
            for r, c in np.argwhere(work == MAROON)
            if int(dist[r, c]) < 10**9
        ]
        if not reachable:
            break

        reachable.sort()
        dmin = reachable[0][0]
        layer = [(r, c) for d, r, c in reachable if d == dmin]
        space = len(DEST_CELLS) - len(collected)
        layer = sorted(layer)[:space]

        for r, c in layer:
            work[r, c] = floor
            collected.append((r, c))

        zones = {_zone_of(cell, route) for cell in layer}
        if stage == 0 and 1 in zones:
            for r, c in route["gate2"]:
                work[r, c] = floor
            opened_segments.append(route["gate2"])
            stage = 1
        elif stage == 1 and 2 in zones:
            for r, c in route["gate3"]:
                work[r, c] = floor
            opened_segments.append(route["gate3"])
            stage = 2

    out = work.copy()
    for r, c in DEST_CELLS[: len(collected)]:
        out[r, c] = MAROON

    return {
        "output": out,
        "route": route,
        "collected": tuple(collected),
        "opened_segments": tuple(opened_segments),
        "initial_reachable": tuple(initial_reachable),
        "floor": floor,
        "wall": wall,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pull maroon pixels into the 3x3 block, but only after they unlock nested wall gates layer by layer.

    Input structure:
    - Orange floor surrounds a fixed green 3x3 destination with one red center.
    - Magenta pixels form three nested rectangular frames. The inner frame has one opening, while the middle and outer frames begin sealed.
    - Maroon pixels appear in three depth bands: one trigger in the first corridor, one trigger in the second corridor, and several more outside the outer frame; extra distractors can remain trapped in opposite sealed corridors.

    Rule:
    - Run BFS from passable cells adjacent to the destination through floor and maroon cells.
    - Collect only the nearest BFS layer of reachable maroons at a time. After the first-annulus layer is collected, a short middle-frame gate disappears; after the second-annulus layer is collected, a short outer-frame gate disappears.
    - Recompute BFS after each wall change, continue layer by layer, erase every moved source to floor, and fill the destination cells in row-major order with the collected maroons.

    Key insight: Reachability is no longer static. The nested frames define a staged escape path, and the visible wall removals prove that later maroons are not initially reachable; they become collectible only because earlier BFS layers crack open the next shell.
    """

    return _simulate(grid)["output"]


def _static_baseline_output(grid: np.ndarray) -> np.ndarray:
    """Original one-shot reachability on the same input, used to reject degenerate generators."""
    grid = np.asarray(grid, dtype=int)
    floor, _wall = _infer_floor_and_wall(grid, DEST_CELLS, DEST_SET)
    passable = {floor, MAROON}
    dist = _bfs_dist(grid, DEST_CELLS, DEST_SET, passable)
    maroons = [
        (int(dist[r, c]), int(r), int(c))
        for r, c in np.argwhere(grid == MAROON)
        if int(dist[r, c]) < 10**9
    ]
    maroons.sort()
    chosen = maroons[: len(DEST_CELLS)]

    out = grid.copy()
    for _d, r, c in chosen:
        out[r, c] = floor
    for r, c in DEST_CELLS[: len(chosen)]:
        out[r, c] = MAROON
    return out


def _base_grid(route: dict) -> np.ndarray:
    grid = np.full((GRID_H, GRID_W), FLOOR_COLOR, dtype=int)
    for r, c in DEST_CELLS:
        grid[r, c] = DEST_COLOR
    grid[CENTER_R, CENTER_C] = RED_CENTER

    wall_cells = BOX1_FRAME | BOX3_FRAME
    wall_cells |= BOX2_WIDE_FRAME if route["orientation"] == "wide" else BOX2_TALL_FRAME
    wall_cells.discard(route["opening"])
    for r, c in wall_cells:
        grid[r, c] = WALL_COLOR
    return grid


def _choose_outer_sources(route: dict) -> list[tuple[int, int]]:
    banned = {route["opening"], route["shell1"], route["shell2"], *route["gate2"], *route["gate3"]}
    pool = [cell for cell in route["outer_pool"] if cell not in banned]
    count = random.randint(*OUTER_COUNT_RANGE)
    return random.sample(pool, count)


def _choose_distractors(route: dict) -> list[tuple[int, int]]:
    count = random.randint(*DISTRACTOR_RANGE)
    count = min(count, len(route["distractors"]))
    return random.sample(list(route["distractors"]), count)


def _fusion_metrics(grid: np.ndarray) -> dict:
    sim = _simulate(grid)
    baseline = _static_baseline_output(grid)
    dynamic_count = int(np.sum(sim["output"][np.array([r for r, _ in DEST_CELLS]), np.array([c for _, c in DEST_CELLS])] == MAROON))
    baseline_count = int(np.sum(baseline[np.array([r for r, _ in DEST_CELLS]), np.array([c for _, c in DEST_CELLS])] == MAROON))
    opened_walls = sum(len(seg) for seg in sim["opened_segments"])
    return {
        "dynamic_count": dynamic_count,
        "baseline_count": baseline_count,
        "delta_count": dynamic_count - baseline_count,
        "opened_walls": opened_walls,
        "initial_reachable": len(sim["initial_reachable"]),
        "route": sim["route"]["name"],
    }


def _gen_attempt() -> np.ndarray | None:
    route = random.choice(ROUTES)
    grid = _base_grid(route)

    sources = [route["shell1"], route["shell2"], *_choose_outer_sources(route)]
    distractors = _choose_distractors(route)
    if len(sources) > len(DEST_CELLS):
        return None
    used = set()
    for cell in sources + distractors:
        if cell in used or cell in DEST_SET:
            return None
        used.add(cell)
        if int(grid[cell]) != FLOOR_COLOR:
            return None
        grid[cell] = MAROON

    try:
        sim = _simulate(grid)
    except Exception:
        return None

    if len(sim["collected"]) != len(sources):
        return None
    if len(sim["opened_segments"]) != 2:
        return None
    if np.array_equal(sim["output"], _static_baseline_output(grid)):
        return None

    metrics = _fusion_metrics(grid)
    if metrics["delta_count"] < 3:
        return None
    if metrics["opened_walls"] != 4:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
