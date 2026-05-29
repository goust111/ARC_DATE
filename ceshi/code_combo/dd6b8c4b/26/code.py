# Combo variant of dd6b8c4b (index 26)
# combo_spec: D2/2+D15/2
# Name: 动态墙壁+BFS逐层收集

from __future__ import annotations

from collections import deque
import random

import numpy as np

from arc_utils import neighbors4


# Design analysis for the fused variant
# ------------------------------------
# a) Concretization and trace:
#    - The original task does one static BFS, sorts every reachable maroon once, then fills the 3x3 target.
#    - This combo turns the wall into a local crumble mechanic: when a maroon is collected, every magenta wall
#      cell in its surrounding 3x3 neighborhood becomes orange floor.
#    - Collection is no longer global. We repeatedly:
#         1. BFS from cells adjacent to the green/red target.
#         2. Take only the currently nearest reachable maroon layer.
#         3. Erase that layer, crumble nearby walls, and recompute BFS.
#      Example in the generator template:
#         hall token -> opens right gate -> right-room token -> opens lower gate -> lower-room tokens.
#      This guarantees a visible collect -> wall disappears -> new room opens -> continue pattern.
# b) Complexity calibration:
#    - Only two dynamic gates are used, but each sample contains multiple layers and 5-8 collected maroons.
#    - The crumble rule is always the same local 3x3 operation, so it is learnable from few examples.
#    - The layout varies by optional filler maroons and grid rotation/flip, which keeps samples from looking cloned
#      without introducing too many independent cases.
# c) Hidden assumptions changed from the original solver:
#    - Static reachability is no longer enough; we must recompute distances after each layer.
#    - Distance tie-breaking only applies within the current layer; later layers depend on earlier wall removals.
#    - A collected maroon may alter the obstacle map, so the solver must mutate a working copy of the grid.
#    - The generator must keep trigger maroons near only their intended gates; otherwise 3x3 crumble would open
#      unrelated walls and make the visual rule messy.
# d) Placement legality:
#    - The destination block is still the fixed 3x3 block centered at (5,5).
#    - All tokens are placed on floor cells outside the destination.
#    - Walls and tokens are laid out in canonical coordinates, then rotated/flipped as a whole, which preserves
#      bounds and prevents overlap.
# e/f/g) Observability and cleanliness:
#    - The disappearing walls remain visible in the final output because they are turned into floor.
#    - Dynamic rooms are separated by full wall segments touching the border or other barriers, so the newly opened
#      area is visually obvious instead of being a subtle connectivity change.
#    - No invisible feature participates in ordering: only BFS distance, row-major order within a layer, and the
#      visible local wall crumble matter.
# h) New solver assumptions and generator guarantees:
#    - There is exactly one red center and at least one floor cell adjacent to the target; the generator guarantees it.
#    - The total collectible maroons stay within 9, so every moved maroon has a destination cell.
#    - Trigger maroons are spaced so that their 3x3 crumble neighborhoods do not accidentally merge unrelated gates.


GRID_H, GRID_W = 11, 11
CENTER_R, CENTER_C = 5, 5
FLOOR_COLOR, WALL_COLOR = 7, 6
DEST_COLOR, RED_CENTER = 3, 2
MAROON = 9
DEST_LIMIT = 9

GEN_RETRIES = 100


def _dest_cells() -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(CENTER_R - 1, CENTER_R + 2) for c in range(CENTER_C - 1, CENTER_C + 2)]
    cells.sort()
    return cells


DEST_CELLS = _dest_cells()
DEST_SET = set(DEST_CELLS)


def _bfs_dist(grid: np.ndarray) -> np.ndarray:
    """Distance from the destination-adjacent frontier through floor/maroon cells; adjacent cells start at 1."""
    h, w = grid.shape
    dist = np.full((h, w), -1, dtype=int)
    q: deque[tuple[int, int]] = deque()

    for r, c in DEST_CELLS:
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in DEST_SET and int(grid[rr, cc]) in (FLOOR_COLOR, MAROON):
                if dist[rr, cc] == -1:
                    dist[rr, cc] = 1
                    q.append((rr, cc))

    assert q, "there must be passable floor next to the destination block"

    while q:
        r, c = q.popleft()
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in DEST_SET:
                if int(grid[rr, cc]) in (FLOOR_COLOR, MAROON) and dist[rr, cc] == -1:
                    dist[rr, cc] = dist[r, c] + 1
                    q.append((rr, cc))

    return dist


def _simulate_dynamic_collection(grid: np.ndarray, *, return_trace: bool = False):
    """
    Run the fused process on a mutable copy of the grid.

    Returns:
    - final_work: grid after maroons are erased and dynamic walls have crumbled
    - collected: maroons collected in layer order, row-major inside each layer
    - trace: per-layer debug information when requested
    """
    work = np.array(grid, dtype=int, copy=True)
    collected: list[tuple[int, int]] = []
    trace: list[dict] = []

    while len(collected) < DEST_LIMIT:
        dist = _bfs_dist(work)
        reachable = [(int(dist[r, c]), int(r), int(c)) for r, c in np.argwhere(work == MAROON) if (r, c) not in DEST_SET and dist[r, c] > 0]
        if not reachable:
            break

        layer_dist = min(d for d, _r, _c in reachable)
        layer = sorted((r, c) for d, r, c in reachable if d == layer_dist)
        room_left = DEST_LIMIT - len(collected)
        layer = layer[:room_left]

        removed_walls: set[tuple[int, int]] = set()
        for r, c in layer:
            collected.append((r, c))
            work[r, c] = FLOOR_COLOR
            for rr in range(r - 1, r + 2):
                for cc in range(c - 1, c + 2):
                    if 0 <= rr < GRID_H and 0 <= cc < GRID_W and int(work[rr, cc]) == WALL_COLOR:
                        removed_walls.add((rr, cc))

        for rr, cc in removed_walls:
            work[rr, cc] = FLOOR_COLOR

        if return_trace:
            trace.append(
                {
                    "distance": layer_dist,
                    "layer": list(layer),
                    "removed_walls": sorted(removed_walls),
                    "reachable_before": len(reachable),
                }
            )

    return work, collected, trace


def _build_output_from_process(inp: np.ndarray):
    final_work, collected, trace = _simulate_dynamic_collection(inp, return_trace=True)
    out = final_work.copy()

    for r, c in DEST_CELLS:
        out[r, c] = DEST_COLOR
    out[CENTER_R, CENTER_C] = RED_CENTER

    for idx, (r, c) in enumerate(DEST_CELLS[: len(collected)]):
        out[r, c] = MAROON

    return out, collected, trace


def _static_reachable_maroons(grid: np.ndarray) -> list[tuple[int, int]]:
    """Reachable maroons under the original static-wall interpretation, used only for generator validation/metrics."""
    dist = _bfs_dist(np.array(grid, dtype=int, copy=True))
    maroons = [(int(r), int(c)) for r, c in np.argwhere(grid == MAROON) if (r, c) not in DEST_SET and dist[r, c] > 0]
    maroons.sort(key=lambda rc: (int(dist[rc[0], rc[1]]), rc[0], rc[1]))
    return maroons[:DEST_LIMIT]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move maroons into the target block by repeatedly taking the current BFS-nearest layer and crumbling nearby walls.

    Input structure:
    - The grid is 11x11 with an orange floor, magenta walls, scattered maroon tokens, and a fixed 3x3 green target
      centered on one red pixel.
    - Some maroons sit in rooms that are initially sealed by magenta barriers, while earlier reachable maroons sit next
      to those barriers.

    Rule:
    - Run BFS from floor/maroon cells adjacent to the target, but collect only the nearest reachable maroon layer.
    - After collecting that layer, erase those maroons to floor and turn every wall in each collected maroon's 3x3
      neighborhood into floor, then recompute BFS on the changed grid.
    - Continue until no more maroons are reachable; place the collected maroons into the target block in row-major order.

    Key insight:
    - Distances are not fixed once at the start: early collections can break nearby walls and open entirely new regions,
      so the correct output comes from an iterative BFS -> collect layer -> crumble walls loop.
    """

    grid = np.asarray(grid, dtype=int)
    centers = np.argwhere(grid == RED_CENTER)
    assert centers.shape == (1, 2), "grid must contain exactly one red center"

    out, _collected, _trace = _build_output_from_process(grid)
    return out


def _paint_dest(grid: np.ndarray) -> None:
    for r, c in DEST_CELLS:
        grid[r, c] = DEST_COLOR
    grid[CENTER_R, CENTER_C] = RED_CENTER


def _paint_inner_ring_walls(grid: np.ndarray) -> None:
    openings = {(3, 4), (3, 5), (3, 6)}
    ring = set()
    for r, c in DEST_CELLS:
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < GRID_H and 0 <= cc < GRID_W and (rr, cc) not in DEST_SET:
                ring.add((rr, cc))
    for r, c in ring:
        if (r, c) not in openings:
            grid[r, c] = WALL_COLOR


def _transform_grid(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, copy=True)
    if random.random() < 0.5:
        out = np.fliplr(out)
    k = random.randint(0, 3)
    out = np.rot90(out, k)
    return np.array(out, dtype=int, copy=False)


def _canonical_input() -> np.ndarray:
    """
    Canonical pre-transform layout:
    - Top hall is initially reachable from the destination.
    - Trigger T1 opens the right room by crumbling a vertical gate.
    - Trigger T2 in the right room opens the lower room by crumbling a horizontal gate.
    """
    grid = np.full((GRID_H, GRID_W), FLOOR_COLOR, dtype=int)
    _paint_dest(grid)
    _paint_inner_ring_walls(grid)

    # Permanent partitions and one sealed distractor pocket.
    for r in range(GRID_H):
        grid[r, 8] = WALL_COLOR
        grid[r, 2] = WALL_COLOR
    for c in range(8, GRID_W):
        grid[5, c] = WALL_COLOR

    # Keep the lower room open internally.
    grid[5, 8] = WALL_COLOR
    grid[5, 9] = WALL_COLOR
    grid[5, 10] = WALL_COLOR

    # Hall / room floors remain orange by default. Mark fixed and optional maroons.
    fixed_tokens = {(3, 5), (2, 7), (4, 9), (7, 9)}

    optional_hall = [(1, 4), (1, 5), (2, 5)]
    optional_right = [(2, 9), (3, 10), (4, 10)]
    optional_lower = [(6, 10), (8, 9), (9, 10)]

    chosen = set(fixed_tokens)
    chosen.update(random.sample(optional_hall, random.randint(1, 2)))
    chosen.update(random.sample(optional_right, 1))
    chosen.update(random.sample(optional_lower, random.randint(1, 2)))

    # Static distractors live left of the full-height wall at column 2.
    distractor_pool = [(7, 0), (8, 1), (9, 0), (10, 1)]
    distractors = random.sample(distractor_pool, random.randint(0, 2))

    for r, c in sorted(chosen):
        assert (r, c) not in DEST_SET and int(grid[r, c]) == FLOOR_COLOR
        grid[r, c] = MAROON
    for r, c in distractors:
        assert int(grid[r, c]) == FLOOR_COLOR
        grid[r, c] = MAROON

    return grid


def _generator_metrics(inp: np.ndarray) -> dict:
    static_reachable = _static_reachable_maroons(inp)
    out, collected, trace = _build_output_from_process(inp)
    del out
    removed_wall_count = sum(len(step["removed_walls"]) for step in trace)
    dynamic_gain = len(collected) - len(static_reachable)
    return {
        "collected": len(collected),
        "static_reachable": len(static_reachable),
        "dynamic_gain": dynamic_gain,
        "removed_wall_count": removed_wall_count,
        "layers": len(trace),
        "opened_layers": sum(1 for step in trace if step["removed_walls"]),
    }


def _gen_attempt() -> np.ndarray | None:
    inp = _transform_grid(_canonical_input())
    metrics = _generator_metrics(inp)

    # Reject accidental degeneration back to the static original behavior.
    if metrics["collected"] < 5:
        return None
    if metrics["dynamic_gain"] < 2:
        return None
    if metrics["removed_wall_count"] < 6:
        return None
    if metrics["layers"] < 4 or metrics["opened_layers"] < 2:
        return None

    # Solver self-check.
    out = main(inp)
    assert out.shape == (GRID_H, GRID_W)
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
