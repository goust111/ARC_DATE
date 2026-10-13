from __future__ import annotations

from collections import deque
from pathlib import Path
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.exists():
    sys.path.append(str(_ARC_UTILS_DIR))

try:
    from arc_utils import in_bounds, neighbors4
except Exception:
    def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
        h, w = shape
        return 0 <= r < h and 0 <= c < w

    def neighbors4(r: int, c: int):
        yield r - 1, c
        yield r + 1, c
        yield r, c - 1
        yield r, c + 1


BLACK = 0
BLUE = 1
RED = 2
GREY = 5
TEAL = 8

PASSABLE = {BLACK, BLUE, RED}
DIRS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}
NODE_DIRS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}


def _blue_seed(grid: np.ndarray) -> tuple[int, int]:
    seeds = np.argwhere(grid == BLUE)
    assert len(seeds) == 1, "expected exactly one blue seed"
    return int(seeds[0, 0]), int(seeds[0, 1])


def _red_rays(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    rays: list[tuple[int, int, int, int]] = []
    for r, c in np.argwhere(grid == RED):
        grey_neighbors = []
        for nr, nc in neighbors4(int(r), int(c)):
            if in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == GREY:
                grey_neighbors.append((nr, nc))
        assert len(grey_neighbors) == 1, "each red must have exactly one grey 4-neighbor"
        nr, nc = grey_neighbors[0]
        rays.append((int(r), int(c), int(nr - r), int(nc - c)))
    return rays


def _erase_grey_rays(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for r, c, dr, dc in _red_rays(grid):
        rr = r + dr
        cc = c + dc
        while in_bounds(out.shape, rr, cc) and int(out[rr, cc]) != TEAL:
            if int(out[rr, cc]) == GREY:
                out[rr, cc] = BLACK
            rr += dr
            cc += dc
    return out


def _reachable_passable(grid: np.ndarray) -> np.ndarray:
    seen = np.zeros(grid.shape, dtype=bool)
    sr, sc = _blue_seed(grid)
    q = deque([(sr, sc)])
    seen[sr, sc] = True
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if not in_bounds(grid.shape, nr, nc) or seen[nr, nc]:
                continue
            if int(grid[nr, nc]) not in PASSABLE:
                continue
            seen[nr, nc] = True
            q.append((nr, nc))
    return seen


def _fill_reachable_corridors(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    reachable = _reachable_passable(out)
    out[(out == BLACK) & reachable] = BLUE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Open ray-aligned grey blockers, then flood-fill the newly connected maze corridors from the blue seed.

    Input structure:
    - A teal maze wall network encloses black one-cell-wide corridors on a fixed `(r, c)` grid.
    - Grey cells are single-cell blockers embedded in straight corridor segments.
    - One blue seed and one to three red operator pixels sit on corridor cells; each red has exactly one grey 4-neighbor.

    Rule:
    - For each red pixel, use its unique grey 4-neighbor to determine a cardinal ray direction.
    - Starting at the red and advancing along that direction, change every grey cell on the ray to black until a teal wall or the grid boundary stops the ray.
    - After all rays are applied, flood through corridor cells using 4-connectivity, treating black, blue, and red corridor cells as passable.
    - Recolor every reachable black corridor cell to blue, while keeping red, teal, and any unerased grey cells unchanged.

    Key insight:
    - Red pixels are not output paint sources; they are directional operators whose ray erasures change the maze topology before the blue flood-fill happens.
    """
    opened = _erase_grey_rays(np.asarray(grid, dtype=int))
    return _fill_reachable_corridors(opened)


def _edge_key(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    return (a, b) if a <= b else (b, a)


def _node_in_bounds(nr: int, nc: int, node: tuple[int, int]) -> bool:
    r, c = node
    return 0 <= r < nr and 0 <= c < nc


def _step_node(node: tuple[int, int], direction: str) -> tuple[int, int]:
    dr, dc = NODE_DIRS[direction]
    return int(node[0] + dr), int(node[1] + dc)


def _node_to_grid(node: tuple[int, int]) -> tuple[int, int]:
    return int(2 * node[0] + 1), int(2 * node[1] + 1)


def _edge_to_grid(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    ar, ac = _node_to_grid(a)
    br, bc = _node_to_grid(b)
    return int((ar + br) // 2), int((ac + bc) // 2)


def _candidate_expansions(
    nr: int,
    nc: int,
    roots: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    *,
    banned_nodes: set[tuple[int, int]] | None = None,
    forbidden_steps: set[tuple[tuple[int, int], str]] | None = None,
) -> list[tuple[tuple[int, int], str, tuple[int, int]]]:
    banned_nodes = banned_nodes or set()
    forbidden_steps = forbidden_steps or set()
    candidates = []
    for node in roots:
        if node in banned_nodes:
            continue
        for direction in DIRS:
            if (node, direction) in forbidden_steps:
                continue
            nxt = _step_node(node, direction)
            if _node_in_bounds(nr, nc, nxt) and nxt not in occupied:
                candidates.append((node, direction, nxt))
    return candidates


def _free_run(
    nr: int,
    nc: int,
    occupied: set[tuple[int, int]],
    start: tuple[int, int],
    direction: str,
) -> list[tuple[int, int]]:
    out = []
    cur = start
    while True:
        cur = _step_node(cur, direction)
        if not _node_in_bounds(nr, nc, cur) or cur in occupied:
            return out
        out.append(cur)


def _grow_base_tree(
    rng: np.random.Generator,
    nr: int,
    nc: int,
    blue_node: tuple[int, int],
) -> tuple[set[tuple[int, int]], dict[tuple[tuple[int, int], tuple[int, int]], int]]:
    target = int(rng.integers(5, min(nr * nc, 11) + 1))
    nodes = {blue_node}
    edges: dict[tuple[tuple[int, int], tuple[int, int]], int] = {}
    while len(nodes) < target:
        candidates = _candidate_expansions(nr, nc, nodes, nodes)
        if not candidates:
            break
        parent, _direction, child = candidates[int(rng.integers(len(candidates)))]
        nodes.add(child)
        edges[_edge_key(parent, child)] = BLACK

    extra_pairs = []
    for node in nodes:
        for direction in DIRS:
            nxt = _step_node(node, direction)
            key = _edge_key(node, nxt)
            if _node_in_bounds(nr, nc, nxt) and nxt in nodes and key not in edges:
                extra_pairs.append((node, nxt))
    rng.shuffle(extra_pairs)
    for node, nxt in extra_pairs[: int(rng.integers(0, 3))]:
        edges[_edge_key(node, nxt)] = BLACK
    return nodes, edges


def _grow_side_subtree(
    rng: np.random.Generator,
    nr: int,
    nc: int,
    occupied: set[tuple[int, int]],
    edge_colors: dict[tuple[tuple[int, int], tuple[int, int]], int],
    anchors: list[tuple[int, int]],
    max_new_nodes: int,
    *,
    forbidden_steps: set[tuple[tuple[int, int], str]] | None = None,
) -> list[tuple[int, int]]:
    new_nodes: list[tuple[int, int]] = []
    target = int(rng.integers(0, max_new_nodes + 1))
    frontier = set(anchors)
    while len(new_nodes) < target:
        candidates = _candidate_expansions(
            nr,
            nc,
            frontier,
            occupied,
            forbidden_steps=forbidden_steps,
        )
        if not candidates:
            break
        parent, _direction, child = candidates[int(rng.integers(len(candidates)))]
        occupied.add(child)
        frontier.add(child)
        new_nodes.append(child)
        edge_colors[_edge_key(parent, child)] = BLACK
    return new_nodes


def _controlled_gate_sites(
    nr: int,
    nc: int,
    occupied: set[tuple[int, int]],
    base_nodes: set[tuple[int, int]],
    blocked_nodes: set[tuple[int, int]],
) -> list[tuple[tuple[int, int], str, list[tuple[int, int]]]]:
    sites = []
    for node in base_nodes:
        if node in blocked_nodes:
            continue
        for direction in DIRS:
            run = _free_run(nr, nc, occupied, node, direction)
            if len(run) >= 2:
                sites.append((node, direction, run))
    return sites


def _blocked_branch_sites(
    nr: int,
    nc: int,
    occupied: set[tuple[int, int]],
    nodes: set[tuple[int, int]],
    blocked_nodes: set[tuple[int, int]],
    forbidden_steps: set[tuple[tuple[int, int], str]],
) -> list[tuple[tuple[int, int], str, list[tuple[int, int]]]]:
    sites = []
    for node in nodes:
        if node in blocked_nodes:
            continue
        for direction in DIRS:
            if (node, direction) in forbidden_steps:
                continue
            run = _free_run(nr, nc, occupied, node, direction)
            if run:
                sites.append((node, direction, run))
    return sites


def _add_controlled_branch(
    rng: np.random.Generator,
    nr: int,
    nc: int,
    occupied: set[tuple[int, int]],
    edge_colors: dict[tuple[tuple[int, int], tuple[int, int]], int],
    site: tuple[tuple[int, int], str, list[tuple[int, int]]],
) -> tuple[tuple[int, int], str, list[tuple[int, int]]]:
    root, direction, run = site
    trunk_len = int(rng.integers(2, min(len(run), 4) + 1))
    trunk = run[:trunk_len]
    prev = root
    for idx, node in enumerate(trunk):
        occupied.add(node)
        color = GREY if idx == 0 or (idx < trunk_len - 1 and rng.random() < 0.45) else BLACK
        edge_colors[_edge_key(prev, node)] = color
        prev = node

    forbidden = {(trunk[-1], direction)}
    _grow_side_subtree(
        rng,
        nr,
        nc,
        occupied,
        edge_colors,
        trunk,
        max_new_nodes=int(rng.integers(1, 4)),
        forbidden_steps=forbidden,
    )
    return root, direction, trunk


def _add_blocked_branch(
    rng: np.random.Generator,
    nr: int,
    nc: int,
    occupied: set[tuple[int, int]],
    edge_colors: dict[tuple[tuple[int, int], tuple[int, int]], int],
    site: tuple[tuple[int, int], str, list[tuple[int, int]]],
) -> tuple[tuple[int, int], str]:
    root, direction, run = site
    trunk_len = int(rng.integers(1, min(len(run), 3) + 1))
    trunk = run[:trunk_len]
    prev = root
    for idx, node in enumerate(trunk):
        occupied.add(node)
        color = GREY if idx == 0 or (idx > 0 and rng.random() < 0.25) else BLACK
        edge_colors[_edge_key(prev, node)] = color
        prev = node

    forbidden = {(trunk[-1], direction)}
    _grow_side_subtree(
        rng,
        nr,
        nc,
        occupied,
        edge_colors,
        trunk,
        max_new_nodes=int(rng.integers(0, 3)),
        forbidden_steps=forbidden,
    )
    return root, direction


def _materialize_grid(
    nr: int,
    nc: int,
    nodes: set[tuple[int, int]],
    edge_colors: dict[tuple[tuple[int, int], tuple[int, int]], int],
    blue_node: tuple[int, int],
    red_nodes: set[tuple[int, int]],
) -> np.ndarray:
    grid = np.full((2 * nr + 1, 2 * nc + 1), TEAL, dtype=int)
    for node in nodes:
        r, c = _node_to_grid(node)
        grid[r, c] = BLACK
    for (a, b), color in edge_colors.items():
        r, c = _edge_to_grid(a, b)
        grid[r, c] = int(color)
    br, bc = _node_to_grid(blue_node)
    grid[br, bc] = BLUE
    for node in red_nodes:
        rr, rc = _node_to_grid(node)
        grid[rr, rc] = RED
    return grid


def _grey_has_corridor_shape(grid: np.ndarray, r: int, c: int) -> bool:
    vertical = (
        in_bounds(grid.shape, r - 1, c)
        and in_bounds(grid.shape, r + 1, c)
        and int(grid[r - 1, c]) in PASSABLE
        and int(grid[r + 1, c]) in PASSABLE
        and in_bounds(grid.shape, r, c - 1)
        and in_bounds(grid.shape, r, c + 1)
        and int(grid[r, c - 1]) == TEAL
        and int(grid[r, c + 1]) == TEAL
    )
    horizontal = (
        in_bounds(grid.shape, r, c - 1)
        and in_bounds(grid.shape, r, c + 1)
        and int(grid[r, c - 1]) in PASSABLE
        and int(grid[r, c + 1]) in PASSABLE
        and in_bounds(grid.shape, r - 1, c)
        and in_bounds(grid.shape, r + 1, c)
        and int(grid[r - 1, c]) == TEAL
        and int(grid[r + 1, c]) == TEAL
    )
    return vertical or horizontal


def _validate_grid(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if np.count_nonzero(grid == BLUE) != 1:
        return False

    red_cells = np.argwhere(grid == RED)
    if not (1 <= len(red_cells) <= 3):
        return False
    for r, c in red_cells:
        grey_neighbors = 0
        for nr, nc in neighbors4(int(r), int(c)):
            if in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == GREY:
                grey_neighbors += 1
        if grey_neighbors != 1:
            return False

    grey_cells = np.argwhere(grid == GREY)
    if len(grey_cells) < len(red_cells):
        return False
    if not all(_grey_has_corridor_shape(grid, int(r), int(c)) for r, c in grey_cells):
        return False

    before = _reachable_passable(grid)
    opened = _erase_grey_rays(grid)
    after = _reachable_passable(opened)
    if int(after.sum()) <= int(before.sum()):
        return False

    for r, c in red_cells:
        if not after[int(r), int(c)]:
            return False

    out = main(grid)
    if np.count_nonzero(out == BLUE) <= np.count_nonzero(grid == BLUE):
        return False
    if np.array_equal(grid, out):
        return False
    return True


def _generate_candidate(rng: np.random.Generator) -> np.ndarray | None:
    nr = int(rng.integers(5, 9))
    nc = int(rng.integers(5, 9))
    blue_node = (int(rng.integers(nr)), int(rng.integers(nc)))

    base_nodes, edge_colors = _grow_base_tree(rng, nr, nc, blue_node)
    occupied = set(base_nodes)

    blocked_for_red = {blue_node}
    red_nodes: set[tuple[int, int]] = set()
    controlled_rays: list[tuple[tuple[int, int], str, list[tuple[int, int]]]] = []

    sites = _controlled_gate_sites(nr, nc, occupied, base_nodes, blocked_for_red)
    distinct_roots = sorted({node for node, _direction, _run in sites})
    if not distinct_roots:
        return None
    possible_red_counts = [k for k in (1, 2, 3) if len(distinct_roots) >= k]
    if not possible_red_counts:
        return None
    red_target = int(rng.choice(possible_red_counts))

    for _ in range(red_target):
        sites = _controlled_gate_sites(nr, nc, occupied, base_nodes, blocked_for_red)
        if not sites:
            return None
        site = sites[int(rng.integers(len(sites)))]
        root, direction, trunk = _add_controlled_branch(rng, nr, nc, occupied, edge_colors, site)
        blocked_for_red.add(root)
        red_nodes.add(root)
        controlled_rays.append((root, direction, trunk))

    forbidden_blocked_steps = {(root, direction) for root, direction, _trunk in controlled_rays}
    blocked_only_count = int(rng.choice([0, 1, 1, 2]))
    blocked_only_used = 0
    blocked_nodes = set(red_nodes) | {blue_node}
    while blocked_only_used < blocked_only_count:
        sites = _blocked_branch_sites(
            nr,
            nc,
            occupied,
            occupied,
            blocked_nodes,
            forbidden_blocked_steps,
        )
        if not sites:
            break
        site = sites[int(rng.integers(len(sites)))]
        root, direction = _add_blocked_branch(rng, nr, nc, occupied, edge_colors, site)
        blocked_nodes.add(root)
        forbidden_blocked_steps.add((root, direction))
        blocked_only_used += 1

    grid = _materialize_grid(nr, nc, occupied, edge_colors, blue_node, red_nodes)
    if not _validate_grid(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        rng = np.random.default_rng()
        grid = _generate_candidate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid maze-ray input")
