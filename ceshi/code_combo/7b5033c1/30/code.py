# Combo variant of 7b5033c1 (index 30)
# combo_spec: D3/0+D4/1
# Name: 树形路径+渐变颜色

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np


ROOT = Path(__file__).resolve()
ARC_UTILS_DIR = None
for parent in ROOT.parents:
    candidate = parent / "arc2_solver_select"
    if candidate.exists():
        ARC_UTILS_DIR = candidate
        break
if ARC_UTILS_DIR is None:
    raise RuntimeError("Could not locate arc2_solver_select for arc_utils import")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import get_bg


# Design analysis for the fused variant
# a) Concrete behavior and trace:
#    - Original task: one thin snake with two endpoints; `main()` picked the top-left endpoint and
#      walked the unique chain, so geometry alone fixed the output order.
#    - Fused task: the object is now one thin tree with a unique hub and 3 or 4 branches. Each
#      branch is bent once into its own quadrant, so the topology reads as hub + arms rather than
#      a staircase snake.
#    - Color is no longer decorative. All branches share the same ascending palette cycle, but each
#      branch starts from a different phase of that cycle. Example with palette [2, 4, 7]:
#         north branch colors from hub outward = [7, 4, 2, 7, 4]
#         east  branch colors from hub outward = [4, 7, 2, 4, 7]
#         south branch colors from hub outward = [2, 4, 7, 2, 4]
#      The correct traversal for north is therefore endpoint -> hub-adjacent, because only
#      [2, 4, 7, 2, 4] matches a forward ascending cycle. Geometry chooses branch order
#      (north/east/south/west), while color chooses each branch's internal direction.
#    - Output is a single column containing only the branch pixels, concatenated clockwise. The hub
#      is a routing marker in the input, not part of any branch color strip in the output.
# b) Complexity calibration:
#    - If every branch were monochrome, D4 would disappear. If every branch had a different random
#      palette, humans would have no repeated support. The design therefore uses one shared cycle of
#      3 or 4 colors repeated across all branches, with unique start phases per branch.
#    - Each branch length is 5-8, so the cycle repeats enough times to make forward vs reverse easy
#      to see. A one-step or two-step branch would make the direction cue too weak.
# c) Broken assumptions from the original solver:
#    - `ends = [p for p in pts if deg[p] == 1]` no longer identifies a single chain; there are now
#      3 or 4 endpoints.
#    - Carrying one `(prev, cur)` state is no longer enough because the walk must split at the hub.
#    - "Top-left endpoint starts the answer" is false. Even once a branch is isolated, the correct
#      read direction comes from its color cycle, not from position.
#    - The new solver therefore assumes: exactly one hub has degree 3 or 4; every non-hub node has
#      degree <= 2; removing the hub yields simple chains; and each branch matches the shared cycle
#      much better in one direction than in the other.
# d) Placement legality:
#    - The generator fixes the hub near the center, assigns each branch a unique cardinal exit, and
#      bends it into a direction-specific quadrant. This prevents overlap, out-of-bounds placement,
#      and accidental side-touching between branches.
#    - After placement it validates connectivity, degrees, color-cycle distinctness, and
#      output-first correctness with `main(input) == expected`.
# e) Visual observability:
#    - Across examples, a human sees the same hub-and-branch silhouette every time, and also sees
#      that each arm carries a repeated color rhythm whose forward direction is not always
#      hub-to-tip. The output groups colors by branch order, so both the tree topology and the
#      direction cue are visible in the input/output pair.
# f) Visual cleanliness:
#    - Branches stay thin, well separated, and only bend once. No extra markers are overlaid on top
#      of the branches, so there is no ambiguity about which pixels belong to which arm.
# g) Color-rule inferability:
#    - Only 3 or 4 branch colors are used, reused across all branches. The structural hint is the
#      cycle itself: branch order comes from position around the hub, while traversal direction is
#      the one that preserves the ascending repeating cycle. The generator forces at least one
#      branch to be reversed relative to hub-to-tip geometry so the color cue is genuinely needed.


Coord = tuple[int, int]
DIRS_CLOCKWISE: list[Coord] = [(-1, 0), (0, 1), (1, 0), (0, -1)]
DIRS4: tuple[Coord, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _neighbors4(cell: Coord) -> list[Coord]:
    r, c = cell
    return [(r + dr, c + dc) for dr, dc in DIRS4]


def _degrees(pset: set[Coord]) -> dict[Coord, int]:
    return {p: sum(1 for q in _neighbors4(p) if q in pset) for p in pset}


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


def _trace_branch(root: Coord, hub: Coord, pset: set[Coord]) -> list[Coord]:
    order: list[Coord] = []
    prev = hub
    cur = root
    seen: set[Coord] = set()
    while True:
        if cur in seen:
            raise ValueError("Branch is not a simple path")
        seen.add(cur)
        order.append(cur)
        nxts = [q for q in _neighbors4(cur) if q in pset and q != prev]
        if not nxts:
            break
        if len(nxts) != 1:
            raise ValueError("Non-hub branch node has multiple forward neighbors")
        prev, cur = cur, nxts[0]
    return order


def _find_hub_and_branches(grid: np.ndarray) -> tuple[int, Coord, dict[Coord, list[Coord]]]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    pts = [tuple(map(int, p)) for p in np.argwhere(g != bg)]
    if not pts:
        return bg, (-1, -1), {}

    pset = set(pts)
    deg = _degrees(pset)
    hubs = [p for p in pts if deg[p] >= 3]
    if len(hubs) != 1:
        raise ValueError("Expected exactly one hub")
    hub = hubs[0]
    if deg[hub] not in (3, 4):
        raise ValueError("Hub degree must be 3 or 4")
    if any(deg[p] > 2 for p in pts if p != hub):
        raise ValueError("Only the hub may branch")

    branches: dict[Coord, list[Coord]] = {}
    for direction in DIRS_CLOCKWISE:
        root = (hub[0] + direction[0], hub[1] + direction[1])
        if root not in pset:
            continue
        branches[direction] = _trace_branch(root, hub, pset)
    if len(branches) != deg[hub]:
        raise ValueError("Branch count does not match hub degree")
    return bg, hub, branches


def _cycle_mismatch_count(colors: list[int], palette: list[int]) -> int:
    period = len(palette)
    best = len(colors) + 1
    for shift in range(period):
        mismatch = 0
        for i, col in enumerate(colors):
            if col != palette[(shift + i) % period]:
                mismatch += 1
        if mismatch < best:
            best = mismatch
    return best


def _choose_branch_read(colors_hub_to_tip: list[int], palette: list[int]) -> list[int]:
    reverse = list(reversed(colors_hub_to_tip))
    forward_score = _cycle_mismatch_count(colors_hub_to_tip, palette)
    reverse_score = _cycle_mismatch_count(reverse, palette)
    if reverse_score < forward_score:
        return reverse
    return colors_hub_to_tip


def _clockwise_geometric_output(grid: np.ndarray) -> list[int]:
    g = np.array(grid, dtype=int)
    _, _, branches = _find_hub_and_branches(g)
    colors: list[int] = []
    for direction in DIRS_CLOCKWISE:
        cells = branches.get(direction)
        if cells is None:
            continue
        colors.extend(int(g[r, c]) for r, c in cells)
    return colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Flatten a hub-and-branch tree into a single column.

    Rule:
    - Find the unique hub.
    - Visit branches in clockwise cardinal order: north, east, south, west.
    - For each branch, compare hub->tip against tip->hub and keep the direction whose colors best
      match a cyclic ascending palette shared by the branches.
    - Concatenate only the branch pixels; the hub itself is omitted from the output strip.
    """
    g = np.array(grid, dtype=int)
    _, _, branches = _find_hub_and_branches(g)
    if not branches:
        return np.zeros((0, 1), dtype=int)

    branch_palette = sorted({int(g[r, c]) for cells in branches.values() for r, c in cells})
    if len(branch_palette) < 3:
        raise ValueError("Expected a shared branch palette of size at least 3")

    out: list[int] = []
    for direction in DIRS_CLOCKWISE:
        cells = branches.get(direction)
        if cells is None:
            continue
        colors = [int(g[r, c]) for r, c in cells]
        out.extend(_choose_branch_read(colors, branch_palette))
    return np.array(out, dtype=int).reshape((-1, 1))


def _build_branch(hub: Coord, direction: Coord, straight_len: int, bend_len: int) -> list[Coord]:
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


def _cycle_sequence(length: int, palette: list[int], shift: int) -> list[int]:
    return [palette[(shift + i) % len(palette)] for i in range(length)]


def _sample_branch_lengths(branch_count: int) -> tuple[int, int]:
    if branch_count == 3:
        return random.randint(3, 4), random.randint(3, 4)
    return random.randint(3, 4), random.randint(2, 3)


def _gen_attempt() -> np.ndarray | None:
    H, W = 17, 17
    hub = (H // 2, W // 2)

    bg = random.randint(0, 9)
    branch_count = random.choice([3, 4])
    selected_dirs = [(-1, 0)] + random.sample([(0, 1), (1, 0), (0, -1)], branch_count - 1)

    branch_cells: dict[Coord, list[Coord]] = {}
    total = 1
    for direction in selected_dirs:
        straight_len, bend_len = _sample_branch_lengths(branch_count)
        cells = _build_branch(hub, direction, straight_len, bend_len)
        branch_cells[direction] = cells
        total += len(cells)

    if total < 20 or total > 30:
        return None

    pset = {hub}
    for cells in branch_cells.values():
        for r, c in cells:
            if not (1 <= r < H - 1 and 1 <= c < W - 1):
                return None
            if (r, c) in pset:
                return None
            pset.add((r, c))

    if not _is_connected(pset):
        return None

    deg = _degrees(pset)
    if [p for p in pset if deg[p] >= 3] != [hub]:
        return None
    if deg[hub] != branch_count:
        return None
    if any(deg[p] > 2 for p in pset if p != hub):
        return None

    usable = [c for c in range(10) if c != bg]
    palette = sorted(random.sample(usable, branch_count))
    hub_choices = [c for c in usable if c not in palette]
    if not hub_choices:
        return None
    hub_color = random.choice(hub_choices)

    offsets = list(range(branch_count))
    random.shuffle(offsets)
    directions_are_reversed = random.sample(selected_dirs, random.randint(1, branch_count - 1))
    reversed_set = set(directions_are_reversed)

    grid = np.full((H, W), bg, dtype=int)
    grid[hub] = hub_color
    expected: list[int] = []

    for idx, direction in enumerate(selected_dirs):
        cells = branch_cells[direction]
        seq = _cycle_sequence(len(cells), palette, offsets[idx])
        placement_cells = list(reversed(cells)) if direction in reversed_set else cells
        for (r, c), col in zip(placement_cells, seq):
            grid[r, c] = col

    for direction in DIRS_CLOCKWISE:
        cells = branch_cells.get(direction)
        if cells is None:
            continue
        colors_hub_to_tip = [int(grid[r, c]) for r, c in cells]
        expected.extend(_choose_branch_read(colors_hub_to_tip, palette))

    expected_arr = np.array(expected, dtype=int).reshape((-1, 1))
    actual_out = main(grid)
    if not np.array_equal(actual_out, expected_arr):
        return None

    geometric_out = np.array(_clockwise_geometric_output(grid), dtype=int).reshape((-1, 1))
    if np.array_equal(geometric_out, expected_arr):
        return None

    start_colors = []
    for direction in DIRS_CLOCKWISE:
        cells = branch_cells.get(direction)
        if cells is None:
            continue
        colors_hub_to_tip = [int(grid[r, c]) for r, c in cells]
        chosen = _choose_branch_read(colors_hub_to_tip, palette)
        start_colors.append(chosen[0])
    if len(start_colors) != len(set(start_colors)):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate fused hub-and-gradient input after 500 attempts")
