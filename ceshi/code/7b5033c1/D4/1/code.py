# Variant of 7b5033c1
# Dimension: D4 — 路径颜色结构 (src)
# Change: 路径分为3-6个连续同色段，每段2-9像素，使用3-5种不同非背景色，相邻段颜色不同... → 渐变颜色序列

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
ARC_UTILS_DIR = ROOT / "arc2_solver_select"
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import get_bg


# Design analysis for the D4 variant
# - Concrete form: the non-background cells still form one simple 4-connected snake, but the
#   intended reading direction is no longer "top-left endpoint first". Instead, the correct
#   endpoint is the one from which the colors read as a repeated ascending palette cycle such as
#   2,4,7,2,4,7,... . Reading from the opposite endpoint produces the reverse cycle 7,4,2,7,4,2,...
#   and therefore a different output strip.
# - Behavior trace: the original solver did `start = min(endpoints)` and then traced geometry.
#   Here we trace from both endpoints, compare both color sequences against the expected ascending
#   cycle over the path's unique colors, and keep the better match. Example:
#   path colors along one endpoint = [2,4,7,2,4,7], reverse = [7,4,2,7,4,2].
#   Original logic would choose whichever endpoint is top-left; the variant must choose the first
#   sequence because only it follows the ascending cycle.
# - Complexity calibration: the generator uses 3-5 distinct colors and total path length 20-30 so
#   the cycle repeats many times. That makes the forward/backward distinction visually obvious.
#   Using only 2 colors or only one short cycle would make the new cue too weak.
# - Hidden assumptions in `main()`: the path is still a single chain with exactly two endpoints;
#   every non-background color belongs to that path; and exactly one endpoint traversal matches the
#   ascending cycle well. The generator enforces all of these. When this assumption does not hold,
#   `main()` falls back to the old top-left rule instead of crashing.
# - Placement legality: `_try_extend` keeps new cells away from the border and rejects any addition
#   that would create an unintended extra 4-neighbor, so the placed object stays in-bounds, has no
#   overlaps, and remains a simple path that the solver can trace.
# - Visual observability: across examples, the output always starts at the cycle head (the smallest
#   palette color) even when that endpoint is not top-left in the input. Humans can see that the
#   right output direction is the one that preserves the repeating color rhythm.

DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _path_neighbors(cell: tuple[int, int], pset: set[tuple[int, int]]) -> list[tuple[int, int]]:
    r, c = cell
    return [(r + dr, c + dc) for dr, dc in DIRS4 if (r + dr, c + dc) in pset]


def _trace_path(start: tuple[int, int], pset: set[tuple[int, int]]) -> list[tuple[int, int]]:
    order: list[tuple[int, int]] = []
    prev: tuple[int, int] | None = None
    cur = start

    for _ in range(len(pset)):
        order.append(cur)
        nxts = [q for q in _path_neighbors(cur, pset) if q != prev]
        if not nxts:
            break
        if len(nxts) != 1:
            raise ValueError("Path is not a simple chain")
        prev, cur = cur, nxts[0]

    if len(order) != len(pset):
        raise ValueError("Failed to trace the full path")
    return order


def _cycle_mismatch_count(colors: list[int], palette: list[int]) -> int:
    period = len(palette)
    return sum(int(col != palette[i % period]) for i, col in enumerate(colors))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Straighten the snake into a single column, choosing the endpoint whose traversal matches
    the ascending repeating palette cycle carried by the path colors.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    pts = [(int(r), int(c)) for r, c in np.argwhere(g != bg)]
    if not pts:
        return np.zeros((0, 1), dtype=int)

    pset = set(pts)
    deg = {p: len(_path_neighbors(p, pset)) for p in pts}
    ends = [p for p in pts if deg[p] == 1]
    if len(ends) != 2:
        raise ValueError("Expected exactly one snake path with two endpoints")

    palette = sorted({int(g[r, c]) for r, c in pts})

    traced_orders = [_trace_path(end, pset) for end in ends]
    traced_colors = [[int(g[r, c]) for r, c in order] for order in traced_orders]
    mismatch_scores = [_cycle_mismatch_count(colors, palette) for colors in traced_colors]

    if mismatch_scores[0] < mismatch_scores[1]:
        chosen = traced_colors[0]
    elif mismatch_scores[1] < mismatch_scores[0]:
        chosen = traced_colors[1]
    else:
        top_left_start = min(ends)
        chosen = traced_colors[ends.index(top_left_start)]

    return np.array(chosen, dtype=int).reshape((-1, 1))


def _try_extend(
    path_set: set[tuple[int, int]],
    last: tuple[int, int],
    dr: int,
    dc: int,
    count: int,
    H: int,
    W: int,
) -> list[tuple[int, int]]:
    """Try to extend the path by `count` steps in direction (dr,dc)."""
    segment: list[tuple[int, int]] = []
    r, c = last
    for i in range(1, count + 1):
        nr, nc = r + dr * i, c + dc * i
        if nr < 1 or nr >= H - 1 or nc < 1 or nc >= W - 1:
            break
        if (nr, nc) in path_set:
            break

        pred = (r + dr * (i - 1), c + dc * (i - 1))
        bad = False
        for ddr, ddc in DIRS4:
            nb = (nr + ddr, nc + ddc)
            if nb != pred and nb in path_set:
                bad = True
                break
        if bad:
            break

        segment.append((nr, nc))
        path_set.add((nr, nc))
    return segment


def _build_cyclic_colors(length: int, palette: list[int]) -> list[int]:
    return [palette[i % len(palette)] for i in range(length)]


def _gen_attempt() -> np.ndarray | None:
    H, W = 16, 16
    bg = random.randint(0, 9)
    palette_size = random.randint(3, 5)
    palette = sorted(random.sample([c for c in range(10) if c != bg], palette_size))

    target_len = random.randint(20, 30)
    start_r = random.randint(1, 4)
    start_c = random.randint(1, W - 2)

    path = [(start_r, start_c)]
    path_set = {(start_r, start_c)}
    turns = 0
    is_vertical = True
    h_dir = random.choice([-1, 1])

    while len(path) < target_len:
        run_len = min(random.randint(1, 5), target_len - len(path))
        if run_len <= 0:
            break

        last = path[-1]
        if is_vertical:
            v_dir = 1 if random.random() < 0.85 else -1
            segment = _try_extend(path_set, last, v_dir, 0, run_len, H, W)
            if not segment:
                segment = _try_extend(path_set, last, -v_dir, 0, run_len, H, W)
        else:
            segment = _try_extend(path_set, last, 0, h_dir, run_len, H, W)
            if not segment:
                segment = _try_extend(path_set, last, 0, -h_dir, run_len, H, W)
                if segment:
                    h_dir = -h_dir
            h_dir = -h_dir

        if not segment:
            break

        path.extend(segment)
        turns += 1
        is_vertical = not is_vertical

    total = len(path)
    if total < 20 or total > 30:
        return None
    if turns < 7 or turns > 11:
        return None

    endpoints = [path[0], path[-1]]
    top_left = min(endpoints)
    true_start = endpoints[0] if endpoints[1] == top_left else endpoints[1]
    oriented_path = path if path[0] == true_start else list(reversed(path))
    colors = _build_cyclic_colors(total, palette)

    grid = np.full((H, W), bg, dtype=int)
    for (r, c), col in zip(oriented_path, colors):
        grid[r, c] = col

    expected_out = np.array(colors, dtype=int).reshape(-1, 1)
    actual_out = main(grid)
    if not np.array_equal(actual_out, expected_out):
        return None

    # Reject accidental degeneracy: the old top-left traversal must disagree with the new output.
    pset = set(path)
    top_left_order = _trace_path(top_left, pset)
    top_left_out = np.array([int(grid[r, c]) for r, c in top_left_order], dtype=int).reshape((-1, 1))
    if np.array_equal(top_left_out, expected_out):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate input after 300 attempts")
