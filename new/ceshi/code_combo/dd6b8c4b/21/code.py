# Combo variant of dd6b8c4b (index 21)
# combo_spec: D1/3+D9/2
# Name: 沿边界追踪可达性+迷宫式墙壁

import random
from collections import deque

import numpy as np

from arc_utils import neighbors4


# Design analysis for the fused variant:
# 1. Concrete behavior: the original task flood-filled all floor reachable from the 3x3 destination block.
#    Here the block has exactly one opening into a one-cell-wide maze, and the solver must walk that maze
#    with a left-hand rule. Maroons on the walked route move inward; maroons in side branches stay put.
# 2. Complexity calibration: a full random maze would be visually noisy and harder to validate, so the
#    generator uses a coarse 5x5 maze lattice mapped to odd cells of the 11x11 canvas. This keeps the maze
#    visibly winding while limiting the degrees of freedom enough for reliable generation.
# 3. Broken original assumptions: the old solver assumed "all passable cells in the same component count".
#    That is no longer true. The new solver therefore assumes exactly one opening next to the destination,
#    exactly one border exit, and that left-hand tracing reaches that exit before ambiguity matters.
# 4. Placement legality: the coarse-lattice renderer never writes outside the 11x11 board, never overlaps
#    the fixed 3x3 destination block, and only carves floor cells where the maze graph explicitly allows it.
# 5. Visual observability: every valid sample includes at least one maroon in a branch that BFS could reach
#    but the left-hand trace does not visit before exiting. This makes the fused rule visibly different from
#    the original flood-fill task from only a few examples.


GRID_H = GRID_W = 11
CENTER_R = CENTER_C = 5
FLOOR_COLOR = 7
WALL_COLOR = 6
DEST_COLOR = 3
RED_CENTER = 2
MAROON = 9

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # N, E, S, W
START_NODES = [(1, 2), (2, 3), (3, 2), (2, 1)]  # top, right, bottom, left
MIN_MAIN_LEN = 6
BRANCH_TARGET_RANGE = (1, 3)
BRANCH_LEN_RANGE = (1, 3)
FILL_K_RANGE = (2, 7)  # leave room for branch distractors so flood-fill would differ
DISTRACTOR_RANGE = (1, 3)
GEN_RETRIES = 400


def _dest_cells() -> list[tuple[int, int]]:
    return [(r, c) for r in range(CENTER_R - 1, CENTER_R + 2) for c in range(CENTER_C - 1, CENTER_C + 2)]


DEST_CELLS = _dest_cells()
DEST_SET = set(DEST_CELLS)


def _all_nodes() -> list[tuple[int, int]]:
    return [(r, c) for r in range(5) for c in range(5) if (r, c) != (2, 2)]


ALL_NODES = _all_nodes()


def _coarse_to_grid(node: tuple[int, int]) -> tuple[int, int]:
    r, c = node
    return 2 * r + 1, 2 * c + 1


def _coarse_neighbors(node: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = node
    out = []
    for dr, dc in DIRS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < 5 and 0 <= nc < 5 and (nr, nc) != (2, 2):
            out.append((nr, nc))
    return out


def _edge_cell(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    ar, ac = _coarse_to_grid(a)
    br, bc = _coarse_to_grid(b)
    return (ar + br) // 2, (ac + bc) // 2


def _is_border_node(node: tuple[int, int]) -> bool:
    r, c = node
    return r in (0, 4) or c in (0, 4)


def _border_exit_cell(node: tuple[int, int]) -> tuple[int, int]:
    gr, gc = _coarse_to_grid(node)
    nr, nc = node
    if nr == 0:
        return 0, gc
    if nr == 4:
        return GRID_H - 1, gc
    if nc == 0:
        return gr, 0
    return gr, GRID_W - 1


def _start_dir(start: tuple[int, int]) -> int:
    r, c = start
    if r < CENTER_R:
        return 0
    if c > CENTER_C:
        return 1
    if r > CENTER_R:
        return 2
    return 3


def _outward_neighbor(node: tuple[int, int]) -> tuple[int, int]:
    r, c = node
    if r == 1 and c == 2:
        return 0, 2
    if r == 2 and c == 3:
        return 2, 4
    if r == 3 and c == 2:
        return 4, 2
    return 2, 0


def _walkable(v: int) -> bool:
    return int(v) in (FLOOR_COLOR, MAROON)


def _find_start(grid: np.ndarray) -> tuple[int, int]:
    starts = set()
    for r, c in DEST_CELLS:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in DEST_SET and _walkable(int(grid[nr, nc])):
                starts.add((nr, nc))
    assert len(starts) == 1, "maze must expose exactly one passable opening next to the destination"
    return next(iter(starts))


def _trace_path(grid: np.ndarray, start: tuple[int, int]) -> tuple[list[tuple[int, int]], bool]:
    cur = start
    facing = _start_dir(start)
    path = [cur]
    step_cap = GRID_H * GRID_W * 8

    for _ in range(step_cap):
        if cur != start and (cur[0] in (0, GRID_H - 1) or cur[1] in (0, GRID_W - 1)):
            return path, True

        moved = False
        for turn in (-1, 0, 1, 2):
            nd = (facing + turn) % 4
            dr, dc = DIRS[nd]
            nr, nc = cur[0] + dr, cur[1] + dc
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and _walkable(int(grid[nr, nc])):
                cur = (nr, nc)
                facing = nd
                path.append(cur)
                moved = True
                break
        if not moved:
            break

    return path, False


def _unique_in_order(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out = []
    seen = set()
    for cell in path:
        if cell not in seen:
            seen.add(cell)
            out.append(cell)
    return out


def _bfs_from_start(grid: np.ndarray, start: tuple[int, int]) -> set[tuple[int, int]]:
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in seen and _walkable(int(grid[nr, nc])):
                seen.add((nr, nc))
                q.append((nr, nc))
    return seen


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pull maroon tokens from the left-hand wall-follow route of a maze into the 3x3 block around the red center.

    Input structure:
    - The 3x3 destination block is fixed at the center: eight green cells around one red center.
    - Outside it, magenta walls carve a one-cell-wide orange maze with exactly one opening touching the block.
    - Maroon tokens lie on maze corridors; some are on the left-hand route and others hide in side branches.

    Rule:
    - Start at the unique corridor cell adjacent to the destination and face away from the center.
    - Follow the maze using a left-first wall-follow order (left, forward, right, back) until the route exits at the border.
    - Collect maroons encountered on that traced route, erase them from their source cells, then fill the destination in row-major order.

    Key insight:
    - Reachability is no longer "everything in the same open region". Only the cells visited by the boundary-following walk count,
      so branch maroons that flood-fill could reach still remain in place unless the left-hand trace actually passes through them.
    """

    grid = np.asarray(grid, dtype=int)
    out = grid.copy()

    centers = np.argwhere(grid == RED_CENTER)
    assert len(centers) == 1, "grid must have exactly one red center"

    start = _find_start(grid)
    path, exited = _trace_path(grid, start)
    assert exited, "left-hand trace must reach a border exit"

    trace_cells = _unique_in_order(path)
    maroons = [(r, c) for r, c in trace_cells if int(grid[r, c]) == MAROON]
    k = min(len(DEST_CELLS), len(maroons))

    for r, c in maroons[:k]:
        out[r, c] = FLOOR_COLOR

    for r, c in DEST_CELLS[:k]:
        out[r, c] = MAROON

    return out


def _random_main_path(
    start: tuple[int, int], end: tuple[int, int], forbidden: set[tuple[int, int]]
) -> list[tuple[int, int]] | None:
    outward = _outward_neighbor(start)
    if end == outward:
        return None

    for _ in range(80):
        path = [start, outward]
        seen = {start, outward}

        def dfs(cur: tuple[int, int]) -> bool:
            if cur == end:
                return len(path) >= MIN_MAIN_LEN

            nbrs = [nb for nb in _coarse_neighbors(cur) if nb not in seen and nb not in forbidden]
            random.shuffle(nbrs)
            if len(path) < MIN_MAIN_LEN - 1 and end in nbrs and len(nbrs) > 1:
                nbrs = [nb for nb in nbrs if nb != end] + [end]

            for nb in nbrs:
                seen.add(nb)
                path.append(nb)
                if dfs(nb):
                    return True
                path.pop()
                seen.remove(nb)
            return False

        if dfs(outward):
            return path
    return None


def _grow_branch(
    used: set[tuple[int, int]], anchor: tuple[int, int], max_len: int, forbidden: set[tuple[int, int]]
) -> list[tuple[int, int]]:
    candidates = [
        nb for nb in _coarse_neighbors(anchor) if nb not in used and nb not in forbidden and not _is_border_node(nb)
    ]
    if not candidates:
        return []

    random.shuffle(candidates)
    cur = candidates[0]
    branch = [anchor, cur]
    local_used = set(used)
    local_used.add(cur)

    while len(branch) - 1 < max_len:
        nxts = [
            nb
            for nb in _coarse_neighbors(cur)
            if nb not in local_used and nb not in forbidden and not _is_border_node(nb)
        ]
        if not nxts or random.random() < 0.35:
            break
        random.shuffle(nxts)
        cur = nxts[0]
        branch.append(cur)
        local_used.add(cur)

    return branch


def _render_structure(main_path: list[tuple[int, int]], branches: list[list[tuple[int, int]]]) -> np.ndarray:
    grid = np.full((GRID_H, GRID_W), WALL_COLOR, dtype=int)
    for r, c in DEST_CELLS:
        grid[r, c] = DEST_COLOR
    grid[CENTER_R, CENTER_C] = RED_CENTER

    corridor = set()
    for seq in [main_path] + branches:
        for node in seq:
            corridor.add(_coarse_to_grid(node))
        for a, b in zip(seq, seq[1:]):
            corridor.add(_edge_cell(a, b))
    corridor.add(_border_exit_cell(main_path[-1]))

    for r, c in corridor:
        if (r, c) not in DEST_SET:
            grid[r, c] = FLOOR_COLOR
    return grid


def _gen_attempt() -> np.ndarray | None:
    start_node = random.choice(START_NODES)
    forbidden = set(START_NODES) - {start_node}
    exit_candidates = [node for node in ALL_NODES if _is_border_node(node) and node != start_node]
    random.shuffle(exit_candidates)

    for exit_node in exit_candidates:
        main_path = _random_main_path(start_node, exit_node, forbidden)
        if main_path is None:
            continue

        used = set(main_path)
        branches: list[list[tuple[int, int]]] = []
        anchors = main_path[1:-1]
        random.shuffle(anchors)
        target_branches = random.randint(*BRANCH_TARGET_RANGE)

        for anchor in anchors:
            if len(branches) >= target_branches:
                break
            branch = _grow_branch(used, anchor, random.randint(*BRANCH_LEN_RANGE), forbidden)
            if len(branch) >= 2:
                branches.append(branch)
                used.update(branch[1:])

        grid = _render_structure(main_path, branches)
        start = _find_start(grid)
        path, exited = _trace_path(grid, start)
        if not exited:
            continue

        trace_cells = _unique_in_order(path)
        trace_set = set(trace_cells)
        bfs_cells = _bfs_from_start(grid, start)
        branch_cells = sorted(cell for cell in bfs_cells - trace_set if cell not in (start, _border_exit_cell(exit_node)))
        source_cells = [cell for cell in trace_cells if cell != start and cell not in DEST_SET]

        if len(branch_cells) < DISTRACTOR_RANGE[0]:
            continue
        if len(source_cells) < FILL_K_RANGE[0]:
            continue

        max_k = min(FILL_K_RANGE[1], len(DEST_CELLS), len(source_cells))
        if max_k < FILL_K_RANGE[0]:
            continue
        k = random.randint(FILL_K_RANGE[0], max_k)

        max_d = min(DISTRACTOR_RANGE[1], len(branch_cells), len(DEST_CELLS) - k)
        if max_d < DISTRACTOR_RANGE[0]:
            continue
        num_d = random.randint(DISTRACTOR_RANGE[0], max_d)

        chosen_sources = source_cells[:k]
        distractors = random.sample(branch_cells, num_d)

        inp = grid.copy()
        for r, c in chosen_sources + distractors:
            inp[r, c] = MAROON

        out = grid.copy()
        for r, c in DEST_CELLS[:k]:
            out[r, c] = MAROON
        for r, c in distractors:
            out[r, c] = MAROON

        try:
            result = main(inp)
        except Exception:
            continue
        if np.array_equal(result, out):
            return inp

    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
