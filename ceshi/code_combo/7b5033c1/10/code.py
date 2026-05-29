# Combo variant of 7b5033c1 (index 10)
# combo_spec: D3/0+D10/0
# Name: 树形路径+图例遍历顺序
#
# Design analysis for the fused variant
# a) Concrete behavior and trace:
#    - Original task: one branch-free snake, two endpoints, trace from the top-left endpoint,
#      then reshape the visited colors into a single column.
#    - Fused task: the main object is now one thin 4-connected tree with a unique hub and
#      3 or 4 bent branches. A separate 2-cell corner legend must be decoded first.
#    - The legend no longer controls row/column shape. Instead:
#         horizontal corner legend -> traverse branches clockwise
#         vertical corner legend   -> traverse branches counterclockwise
#      The output remains a single column.
#    - To keep the order anchored, every generated tree includes a north branch. The hub color
#      is emitted first, then the branch colors are appended from the hub outward. Example:
#         hub=9, north branch=[2,2,2], east=[4,4], south=[7,7,7], west absent
#         horizontal legend => [9,2,2,2,4,4,7,7,7]^T
#         vertical legend   => [9,2,2,2,7,7,7,4,4]^T
#      The difference from the original is that endpoint choice disappears; instead the solver
#      must isolate the legend, identify the hub, and concatenate simple branches in the
#      legend-specified cyclic order.
# b) Complexity calibration:
#    - The tree uses 3 or 4 branches and each branch bends once, so the input is visibly not
#      the original staircase snake and is still easy to segment by eye.
#    - Only two legend states exist (horizontal / vertical), which keeps the symbolic decoding
#      learnable from a few examples.
#    - Branch colors are distinct and branch lengths vary, so clockwise vs counterclockwise
#      produces visibly different outputs rather than a trivial permutation.
# c) Hidden-assumption audit against the original main():
#    - `mask = g != bg` no longer means "all foreground cells belong to one path" because the
#      legend is a disconnected foreground component. The solver must find and remove it first.
#    - Degree-based endpoint detection is invalid: the tree has 3 or 4 leaves, not 2.
#    - The old `(prev, cur)` single-walk loop assumes there is exactly one forward choice.
#      That fails at the hub. The new solver instead traces each hub-adjacent branch separately.
#    - `reshape((-1, 1))` remains valid, but only after replacing the original snake order by
#      a branch-order rule decoded from the legend.
#    - New fused assumptions introduced here:
#         * exactly one size-2 straight component sits in a corner and is the legend
#         * after removing it, the remaining foreground is one tree with one hub
#         * the north root exists, providing a shared cyclic anchor for cw/ccw traversal
#      The generator explicitly constructs and verifies all three.
# d) Placement legality:
#    - The hub is fixed near the center of a 17x17 board.
#    - Each chosen branch exits in a unique cardinal direction and then bends into its own
#      quadrant, preventing overlap and accidental side-touching.
#    - The legend occupies exactly two corner cells and uses colors outside the tree palette,
#      so it stays disconnected and visually separate from the tree.
# e) Visual observability:
#    - A human sees one salient corner token plus one central multi-arm tree. Across examples,
#      flipping the legend orientation flips the branch concatenation order while the output
#      remains a single color column. That makes the fusion rule externally visible.
# f) Visual cleanliness:
#    - No region contains stacked symbolic layers: the tree stays centered and the legend stays
#      in a corner. Bent branches keep enough spacing that the hub-and-branch topology remains
#      readable instead of collapsing into clutter.
# g) Color inferability:
#    - Colors are not remapped. The solver only reorders existing monochrome branch runs.
#      Using one color per branch plus one hub color keeps the pattern inferable from structure.

from __future__ import annotations

import os
import random
import sys

import numpy as np

sys.path.insert(
    0,
    os.path.normpath(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            "..",
            "..",
            "..",
            "arc2_solver_select",
        )
    ),
)
from arc_utils import get_bg


Coord = tuple[int, int]
DIRS_CLOCKWISE: list[Coord] = [(-1, 0), (0, 1), (1, 0), (0, -1)]
DIRS_COUNTERCLOCKWISE: list[Coord] = [(-1, 0), (0, -1), (1, 0), (0, 1)]


def _neighbors4(cell: Coord) -> list[Coord]:
    r, c = cell
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _find_components(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[Coord]] = []

    for r, c in np.argwhere(mask):
        rr, cc = int(r), int(c)
        if seen[rr, cc]:
            continue
        stack = [(rr, cc)]
        seen[rr, cc] = True
        comp: list[Coord] = []
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nr, nc in _neighbors4(cur):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(comp)
    return comps


def _is_corner_legend(comp: list[Coord], h: int, w: int) -> bool:
    if len(comp) != 2:
        return False

    rows = sorted({r for r, _ in comp})
    cols = sorted({c for _, c in comp})

    if len(rows) == 1 and len(cols) == 2 and cols[1] == cols[0] + 1:
        return rows[0] in (0, h - 1) and cols in ([0, 1], [w - 2, w - 1])
    if len(cols) == 1 and len(rows) == 2 and rows[1] == rows[0] + 1:
        return cols[0] in (0, w - 1) and rows in ([0, 1], [h - 2, h - 1])
    return False


def _decode_legend_and_tree(grid: np.ndarray, bg: int) -> tuple[bool, list[Coord]]:
    mask = grid != bg
    comps = _find_components(mask)
    h, w = grid.shape

    legends = [comp for comp in comps if _is_corner_legend(comp, h, w)]
    if len(legends) != 1:
        raise ValueError(f"expected exactly one corner legend, found {len(legends)}")

    legend = legends[0]
    clockwise = len({r for r, _ in legend}) == 1

    tree_mask = mask.copy()
    for r, c in legend:
        tree_mask[r, c] = False

    tree_comps = _find_components(tree_mask)
    if len(tree_comps) != 1:
        raise ValueError(f"expected exactly one tree component, found {len(tree_comps)}")
    return clockwise, tree_comps[0]


def _degrees(pset: set[Coord]) -> dict[Coord, int]:
    return {p: sum(1 for q in _neighbors4(p) if q in pset) for p in pset}


def _trace_branch(root: Coord, hub: Coord, pset: set[Coord]) -> list[Coord]:
    order: list[Coord] = []
    prev = hub
    cur = root
    seen: set[Coord] = set()
    while True:
        if cur in seen:
            raise ValueError("branch is not a simple path")
        seen.add(cur)
        order.append(cur)
        nxts = [q for q in _neighbors4(cur) if q in pset and q != prev]
        if not nxts:
            break
        if len(nxts) != 1:
            raise ValueError("non-hub branch node has multiple forward neighbors")
        prev, cur = cur, nxts[0]
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Flatten a hub-and-branch tree into a single color column.

    Variant rule:
    - A disconnected 2-cell corner legend is decoded first.
    - Horizontal legend means clockwise branch order from the north branch.
    - Vertical legend means counterclockwise branch order from the north branch.
    - After the hub color, every present branch is read from the hub outward.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    clockwise, pts = _decode_legend_and_tree(g, bg)
    if not pts:
        return np.zeros((0, 1), dtype=int)

    pset = set(pts)
    deg = _degrees(pset)
    hubs = [p for p in pts if deg[p] >= 3]
    if len(hubs) != 1:
        raise ValueError(f"expected one hub, found {len(hubs)}")
    hub = hubs[0]
    if deg[hub] not in (3, 4):
        raise ValueError(f"hub degree must be 3 or 4, got {deg[hub]}")
    if any(deg[p] > 2 for p in pts if p != hub):
        raise ValueError("tree contains a second branch point")

    north_root = (hub[0] - 1, hub[1])
    if north_root not in pset:
        raise ValueError("missing north anchor branch")

    traversal_dirs = DIRS_CLOCKWISE if clockwise else DIRS_COUNTERCLOCKWISE
    visited = {hub}
    colors = [int(g[hub])]

    for dr, dc in traversal_dirs:
        root = (hub[0] + dr, hub[1] + dc)
        if root not in pset:
            continue
        branch = _trace_branch(root, hub, pset)
        for cell in branch:
            if cell in visited:
                raise ValueError("branch traversal revisited a cell")
            visited.add(cell)
        colors.extend(int(g[r, c]) for r, c in branch)

    if visited != pset:
        raise ValueError("failed to cover the whole tree")
    return np.array(colors, dtype=int).reshape((-1, 1))


def _build_branch(hub: Coord, direction: Coord, straight_len: int, bend_len: int) -> list[Coord]:
    bend_map = {
        (-1, 0): (0, -1),
        (0, 1): (-1, 0),
        (1, 0): (0, 1),
        (0, -1): (1, 0),
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


def _legend_cells(h: int, w: int, clockwise: bool, corner: str) -> list[Coord]:
    if clockwise:
        if corner == "tl":
            return [(0, 0), (0, 1)]
        if corner == "tr":
            return [(0, w - 2), (0, w - 1)]
        if corner == "bl":
            return [(h - 1, 0), (h - 1, 1)]
        return [(h - 1, w - 2), (h - 1, w - 1)]

    if corner == "tl":
        return [(0, 0), (1, 0)]
    if corner == "tr":
        return [(0, w - 1), (1, w - 1)]
    if corner == "bl":
        return [(h - 2, 0), (h - 1, 0)]
    return [(h - 2, w - 1), (h - 1, w - 1)]


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


def _tree_is_valid(pset: set[Coord], hub: Coord, branch_count: int) -> bool:
    if not _is_connected(pset):
        return False
    deg = _degrees(pset)
    hubs = [p for p in pset if deg[p] >= 3]
    if hubs != [hub]:
        return False
    if deg[hub] != branch_count:
        return False
    if (hub[0] - 1, hub[1]) not in pset:
        return False
    if any(deg[p] > 2 for p in pset if p != hub):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h, w = 17, 17
    hub = (h // 2, w // 2)

    bg = random.randint(0, 9)
    branch_count = random.choice([3, 4])
    selected_dirs = [(-1, 0)] + random.sample([(0, 1), (1, 0), (0, -1)], branch_count - 1)

    branch_cells: dict[Coord, list[Coord]] = {}
    total = 1
    for direction in selected_dirs:
        straight_len = random.randint(3, 5)
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
            if not (1 <= r < h - 1 and 1 <= c < w - 1):
                return None
            if cell in pset:
                return None
            pset.add(cell)

    if not _tree_is_valid(pset, hub, branch_count):
        return None

    usable = [c for c in range(10) if c != bg]
    palette = random.sample(usable, branch_count + 3)
    hub_color = palette[0]
    branch_colors = {direction: palette[i + 1] for i, direction in enumerate(selected_dirs)}
    legend_colors = palette[branch_count + 1 : branch_count + 3]

    clockwise = random.random() < 0.5
    corner = random.choice(["tl", "tr", "bl", "br"])

    grid = np.full((h, w), bg, dtype=int)
    grid[hub] = hub_color
    for direction, cells in branch_cells.items():
        color = branch_colors[direction]
        for r, c in cells:
            grid[r, c] = color

    for (r, c), color in zip(_legend_cells(h, w, clockwise, corner), legend_colors):
        grid[r, c] = color

    expected = [hub_color]
    order_dirs = DIRS_CLOCKWISE if clockwise else DIRS_COUNTERCLOCKWISE
    for direction in order_dirs:
        cells = branch_cells.get(direction)
        if cells is None:
            continue
        expected.extend([branch_colors[direction]] * len(cells))

    expected_arr = np.array(expected, dtype=int).reshape((-1, 1))
    actual = main(grid)
    if not np.array_equal(actual, expected_arr):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate fused hub-and-legend input after 500 attempts")
