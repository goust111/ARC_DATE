# Combo variant of 7b5033c1 (index 27)
# combo_spec: D3/0+D5/2+D10/0
# Name: 树形路径+图例顺序+分行输出
#
# Design analysis for the fused variant
# a) Concrete behavior and trace:
#    - Original task: one simple snake, two endpoints, traverse from the top-left
#      endpoint, then reshape the color sequence into an Nx1 column.
#    - Fused task input: one hub-and-branch tree plus one disconnected 2-cell
#      corner legend. The tree has one hub and 3 or 4 simple branches. Every
#      branch is monochrome and leaves the hub in a cardinal direction before a
#      single bend into its own quadrant. The legend is horizontal or vertical.
#    - Fused task output: remove the legend, find the hub, trace each branch from
#      the hub outward, order branches by a directional sweep anchored at north,
#      and place each branch as one output row. Horizontal legend means clockwise
#      sweep (N,E,S,W). Vertical legend means counterclockwise sweep (N,W,S,E).
#      Shorter rows are padded with background.
#    - Small trace:
#         input branches: N=[2,2,2,2,2], E=[5,5,5], S=[8,8,8,8]
#         horizontal legend -> rows [N,E,S]
#         vertical legend   -> rows [N,S,E]
#      The old solver would fail twice: it would treat the legend as part of the
#      path, and it assumes one linear chain rather than a branched tree.
# b) Complexity calibration:
#    - A plain plus-sign would be too weak, so every branch has a bend and varying
#      length. Branch count is 3 or 4, not arbitrary, which keeps the topology
#      rich enough to be visually distinct but still easy to decode.
#    - The legend has only two orientations, giving a small symbolic vocabulary.
#      The generator forces north to exist so the rotational order has a visible
#      anchor rather than drifting sample to sample.
#    - Output rows have at least two distinct lengths, so the packed multi-row
#      layout is visibly different from the original single-column task.
# c) Implicit-assumption audit against the original main():
#    - `mask = g != bg` no longer isolates the main object, because the legend is
#      also non-background. The fused solver must split components first.
#    - Endpoint logic (`deg == 1`) is no longer enough; the tree has 3 or 4 leaf
#      endpoints and one hub with degree 3 or 4.
#    - The old single `prev, cur` walk cannot cover multiple branches. The new
#      solver traces each branch independently after locating the hub.
#    - `reshape((-1, 1))` depended on the output always being a single column.
#      Here the final layout is a KxM packed table, one row per branch.
#    - New fused assumptions introduced by the new solver:
#         1. Exactly one size-2 straight corner component is the legend.
#         2. Removing that legend leaves exactly one 4-connected tree component.
#         3. That tree has exactly one hub with degree 3 or 4, and every other
#            node has degree at most 2.
#         4. Branches attach directly in cardinal directions, so the clockwise /
#            counterclockwise directional sweep is well-defined.
#      The generator explicitly constructs and validates all four properties.
# d) Placement legality:
#    - The hub is fixed at the center of a 19x19 board. Each branch uses a fixed
#      bend map into its own quadrant, preventing overlap and accidental side
#      touching. Length choices are bounded so branches never reach the borders.
#    - The legend is reserved for a corner and uses colors excluded from the tree.
#      With the tree staying inside a wide margin, the legend remains disconnected.
# e) Visual observability:
#    - Humans see one salient bent tree near the center and one tiny corner mark.
#      Across examples, flipping the legend orientation flips the row order while
#      keeping the same branch shapes, which makes the symbolic step observable.
#    - The output is a compact banded table whose row count matches the number of
#      branches. This makes the D5-style repacking obvious.
# f) Visual cleanliness:
#    - There are only two object families: the central tree and the corner legend.
#      They never overlap, and the legend stays in a border corner far from the
#      bent branches. Within the tree, each branch owns its quadrant, so there is
#      no crowded region where multiple semantics stack on the same cells.
# g) Color-rule inferability:
#    - Colors are not remapped. Each output row keeps the branch's original color.
#      The generator uses distinct colors for hub, branches, and legend so order
#      changes are attributable to structure and the legend, not to hidden color
#      arithmetic. The solver does not rely on color uniqueness, but the generator
#      keeps it for human readability and to avoid degenerate samples.

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/arc2_solver_select")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

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
        seen[rr, cc] = True
        stack = [(rr, cc)]
        comp: list[Coord] = []
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nxt in _neighbors4(cur):
                nr, nc = nxt
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append(nxt)
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
    mask = grid != int(bg)
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


def _extract_branch_rows(grid: np.ndarray, pts: list[Coord], clockwise: bool) -> list[list[int]]:
    pset = set(pts)
    if not pset:
        return []

    deg = _degrees(pset)
    hubs = [p for p in pset if deg[p] >= 3]
    if len(hubs) != 1:
        raise ValueError(f"expected one hub, found {len(hubs)}")
    hub = hubs[0]
    if deg[hub] not in (3, 4):
        raise ValueError(f"hub degree must be 3 or 4, got {deg[hub]}")
    if any(deg[p] > 2 for p in pset if p != hub):
        raise ValueError("tree contains a second branch point")

    dirs = DIRS_CLOCKWISE if clockwise else DIRS_COUNTERCLOCKWISE
    rows: list[list[int]] = []
    seen_cells = {hub}
    for dr, dc in dirs:
        root = (hub[0] + dr, hub[1] + dc)
        if root not in pset:
            continue
        branch = _trace_branch(root, hub, pset)
        seen_cells.update(branch)
        rows.append([int(grid[r, c]) for r, c in branch])

    if seen_cells != pset:
        raise ValueError("directional sweep did not cover the whole tree")
    if not rows:
        raise ValueError("tree has no branches")
    return rows


def _pack_rows(rows: list[list[int]], bg: int) -> np.ndarray:
    if not rows:
        return np.zeros((0, 0), dtype=int)

    out_h = len(rows)
    out_w = max(len(row) for row in rows)
    out = np.full((out_h, out_w), int(bg), dtype=int)
    for r, row in enumerate(rows):
        out[r, : len(row)] = np.array(row, dtype=int)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a corner legend, order hub branches, and pack them into rows.

    Variant rule:
    - Exactly one disconnected 2-cell corner legend is present.
    - Horizontal legend => order branches clockwise from north.
    - Vertical legend => order branches counterclockwise from north.
    - Remove the legend, find the unique hub, trace every branch outward.
    - Output one row per branch, left-aligned, padded with background.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    clockwise, pts = _decode_legend_and_tree(g, bg)
    rows = _extract_branch_rows(g, pts, clockwise)
    return _pack_rows(rows, bg)


def _build_branch(hub: Coord, direction: Coord, straight_len: int, bend_len: int) -> list[Coord]:
    bend_map = {
        (-1, 0): (0, -1),   # north -> NW quadrant
        (0, 1): (-1, 0),    # east -> NE quadrant
        (1, 0): (0, 1),     # south -> SE quadrant
        (0, -1): (1, 0),    # west -> SW quadrant
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


def _branch_lengths(branch_cells: dict[Coord, list[Coord]]) -> list[int]:
    return [len(cells) for cells in branch_cells.values()]


def _expected_output(grid: np.ndarray, pset: set[Coord], clockwise: bool, bg: int) -> np.ndarray:
    rows = _extract_branch_rows(grid, list(pset), clockwise)
    return _pack_rows(rows, bg)


def _gen_attempt() -> np.ndarray | None:
    h, w = 19, 19
    hub = (h // 2, w // 2)

    bg = random.randint(0, 9)
    branch_count = random.choice([3, 4])
    selected_dirs = [(-1, 0)] + random.sample([(0, 1), (1, 0), (0, -1)], branch_count - 1)

    branch_cells: dict[Coord, list[Coord]] = {}
    total_cells = 1
    for direction in selected_dirs:
        straight_len = random.randint(3, 4)
        bend_len = random.randint(2, 4)
        cells = _build_branch(hub, direction, straight_len, bend_len)
        branch_cells[direction] = cells
        total_cells += len(cells)

    lengths = _branch_lengths(branch_cells)
    if total_cells < 20 or total_cells > 30:
        return None
    if len(set(lengths)) < 2:
        return None

    pset = {hub}
    for cells in branch_cells.values():
        for cell in cells:
            r, c = cell
            if not (2 <= r < h - 2 and 2 <= c < w - 2):
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

    usable = [c for c in range(10) if c != bg]
    colors = random.sample(usable, branch_count + 3)
    hub_color = colors[0]
    branch_palette = colors[1 : 1 + branch_count]
    legend_colors = colors[1 + branch_count :]
    branch_colors = {direction: branch_palette[i] for i, direction in enumerate(selected_dirs)}

    clockwise = random.random() < 0.5
    corner = random.choice(["tl", "tr", "bl", "br"])
    legend = _legend_cells(h, w, clockwise, corner)

    grid = np.full((h, w), bg, dtype=int)
    grid[hub] = hub_color
    for direction, cells in branch_cells.items():
        color = branch_colors[direction]
        for r, c in cells:
            grid[r, c] = color
    for (r, c), color in zip(legend, legend_colors):
        grid[r, c] = color

    expected = _expected_output(grid, pset, clockwise, bg)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    opposite = _expected_output(grid, pset, not clockwise, bg)
    if np.array_equal(expected, opposite):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate combo input after 400 attempts")
