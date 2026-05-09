# Variant of 7b5033c1
# Dimension: D3 — 源对象：蛇形路径结构 (src)
# Change: 单条4-连通无分支蛇形路径，20-30像素，阶梯形交替水平/垂直走向，7-11个转弯... → 树形分支路径
#
# Design analysis for the D3 change
# a) Concrete behavior:
#    - The input no longer contains a single chain with two endpoints. Instead it contains
#      one thin 4-connected tree with exactly one hub cell and 3 or 4 branches.
#    - Every branch is monochrome and has its own color; the hub has a separate color.
#    - Each branch leaves the hub in a cardinal direction, then bends once into its own
#      quadrant so the "hub_and_branch" topology is easy to see by eye.
#    - The output stays a single column, but it is now:
#         [hub color] + branch pixels from the north branch, then east, then south, then west
#      (skipping missing directions). Within each branch the pixels are read from the hub outward.
#    - Small trace:
#         Original: top-left endpoint -> walk one unique next cell each step -> single linear order.
#         Variant: identify hub -> split the tree into several branch chains -> order the branches
#         clockwise -> concatenate per-branch outward traversals.
#      The key difference is that the ordering problem moves from "which endpoint starts the chain"
#      to "which branch comes next around the hub".
# b) Complexity calibration:
#    - A plain 3-spoke plus-sign would make the change too weak. The generator therefore gives each
#      branch a visible bend and varying length, while keeping the object thin and unambiguous.
#    - Branch colors are all distinct, so the output visibly groups by branch rather than looking
#      like an arbitrary recoloring of the old snake.
# c) Old implicit assumptions that break:
#    - `ends = [p for p in pts if deg[p] == 1]` used to identify the only two endpoints. In the
#      variant there are 3 or 4 endpoints, so "pick the top-left endpoint" is no longer enough.
#    - The old loop carried a single `(prev, cur)` pair and chose one next neighbor. At a hub this
#      would arbitrarily enter one branch and miss the others.
#    - The old solver therefore relied on the hidden invariant "all non-bg pixels form one chain".
#      The new solver replaces that with "exactly one node has degree >= 3, every non-hub node has
#      degree <= 2, and each component after removing the hub is a simple branch".
# d) Placement legality:
#    - The generator fixes the hub near the center and assigns each branch a unique cardinal exit
#      and a unique bend quadrant. This prevents overlap, out-of-bounds placement, and accidental
#      side-touching between branches.
#    - After construction it validates connectivity and degrees, then verifies `main(input)` matches
#      the output-first target.
# e) Visual observability:
#    - Across samples, humans will see a central hub plus several colored bent arms, not a staircase
#      snake. The output becomes one hub pixel followed by color blocks in clockwise branch order,
#      so the topology change is visible in both input and output.

from __future__ import annotations

import random
from collections import Counter

import numpy as np


Coord = tuple[int, int]
DIRS_CLOCKWISE: list[tuple[int, int]] = [(-1, 0), (0, 1), (1, 0), (0, -1)]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _neighbors4(cell: Coord) -> list[Coord]:
    r, c = cell
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _degrees(pset: set[Coord]) -> dict[Coord, int]:
    return {p: sum(1 for q in _neighbors4(p) if q in pset) for p in pset}


def _trace_branch(root: Coord, hub: Coord, pset: set[Coord]) -> list[Coord]:
    """Trace a single simple branch from the hub-adjacent root to its endpoint."""
    order: list[Coord] = []
    prev = hub
    cur = root
    seen: set[Coord] = set()
    while True:
        if cur in seen:
            raise AssertionError("Branch is not a simple path")
        seen.add(cur)
        order.append(cur)
        nxts = [q for q in _neighbors4(cur) if q in pset and q != prev]
        if not nxts:
            break
        if len(nxts) != 1:
            raise AssertionError("Non-hub branch node has multiple forward neighbors")
        prev, cur = cur, nxts[0]
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Flatten a hub-and-branch tree into a single column.

    Input structure:
    - bg = most common color.
    - Non-bg pixels form one 4-connected tree with exactly one hub (degree 3 or 4).
    - Each branch is a simple chain attached to the hub and has its own color.

    Rule:
    - Find the unique hub cell.
    - Visit branches in clockwise cardinal order: north, east, south, west.
    - Output the hub color once, then append every branch's colors from the hub outward.
    """
    g = np.array(grid, dtype=int)
    bg = _get_bg(g)

    pts = [tuple(map(int, p)) for p in np.argwhere(g != bg)]
    if not pts:
        return np.zeros((0, 1), dtype=int)

    pset = set(pts)
    deg = _degrees(pset)
    hubs = [p for p in pts if deg[p] >= 3]
    assert len(hubs) == 1
    hub = hubs[0]
    assert deg[hub] in (3, 4)
    assert all(deg[p] <= 2 for p in pts if p != hub)

    colors = [int(g[hub])]
    for dr, dc in DIRS_CLOCKWISE:
        root = (hub[0] + dr, hub[1] + dc)
        if root not in pset:
            continue
        branch = _trace_branch(root, hub, pset)
        colors.extend(int(g[r, c]) for r, c in branch)

    assert len(colors) == len(pts)
    return np.array(colors, dtype=int).reshape((-1, 1))


def _build_branch(hub: Coord, direction: tuple[int, int], straight_len: int, bend_len: int) -> list[Coord]:
    bend_map = {
        (-1, 0): (0, -1),   # north branch bends into the NW quadrant
        (0, 1): (-1, 0),    # east branch bends into the NE quadrant
        (1, 0): (0, 1),     # south branch bends into the SE quadrant
        (0, -1): (1, 0),    # west branch bends into the SW quadrant
    }
    dr, dc = direction
    bdr, bdc = bend_map[direction]
    r, c = hub
    cells: list[Coord] = []
    for step in range(1, straight_len + 1):
        cells.append((r + dr * step, c + dc * step))
    br, bc = cells[-1]
    for step in range(1, bend_len + 1):
        cells.append((br + bdr * step, bc + bdc * step))
    return cells


def _is_connected(pset: set[Coord]) -> bool:
    if not pset:
        return True
    start = next(iter(pset))
    seen = {start}
    stack = [start]
    while stack:
        cur = stack.pop()
        for nxt in _neighbors4(cur):
            if nxt in pset and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen == pset


def _gen_attempt() -> np.ndarray | None:
    H, W = 17, 17
    hub = (H // 2, W // 2)

    bg = random.randint(0, 9)
    branch_count = random.choice([3, 4])
    selected_dirs = [(-1, 0)] + random.sample([(0, 1), (1, 0), (0, -1)], branch_count - 1)

    branch_cells: dict[tuple[int, int], list[Coord]] = {}
    total = 1
    for direction in selected_dirs:
        straight_len = random.randint(3, 4)
        bend_len = random.randint(2, 4)
        cells = _build_branch(hub, direction, straight_len, bend_len)
        branch_cells[direction] = cells
        total += len(cells)

    if total < 20 or total > 30:
        return None

    pset = {hub}
    for cells in branch_cells.values():
        for cell in cells:
            r, c = cell
            if not (1 <= r < H - 1 and 1 <= c < W - 1):
                return None
            if cell in pset:
                return None
            pset.add(cell)

    if not _is_connected(pset):
        return None

    deg = _degrees(pset)
    hubs = [p for p in pset if deg[p] >= 3]
    if hubs != [hub]:
        return None
    if deg[hub] != branch_count:
        return None
    if any(deg[p] > 2 for p in pset if p != hub):
        return None

    palette = random.sample([c for c in range(10) if c != bg], branch_count + 1)
    hub_color = palette[0]
    branch_colors = {direction: palette[i + 1] for i, direction in enumerate(selected_dirs)}

    grid = np.full((H, W), bg, dtype=int)
    grid[hub] = hub_color
    expected = [hub_color]
    for direction in DIRS_CLOCKWISE:
        cells = branch_cells.get(direction)
        if cells is None:
            continue
        color = branch_colors[direction]
        for r, c in cells:
            grid[r, c] = color
        expected.extend([color] * len(cells))

    actual = main(grid)
    expected_arr = np.array(expected, dtype=int).reshape((-1, 1))
    if not np.array_equal(actual, expected_arr):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate hub-and-branch input after 400 attempts")
