# Combo variant of 7b5033c1 (index 28)
# combo_spec: D1/0+D4/1
# Name: 折叠矩阵+渐变颜色
#
# Design analysis
# a) Concrete behavior and trace:
#    The input still contains exactly one non-background 4-connected snake path. In the source
#    task, `main()` picked the top-left endpoint, traced the snake, and returned an N x 1 column.
#    In this combo, the correct endpoint is the one whose colors follow an ascending repeating
#    cycle over the snake palette, and the traced 1-D stream is then folded into a matrix whose
#    width equals that cycle length.
#    Example with palette [2, 4, 7]:
#      correct traversal colors = [2, 4, 7, 2, 4, 7, 2, 4, 7]
#      reverse traversal colors = [7, 4, 2, 7, 4, 2, 7, 4, 2]
#      source output = a 9 x 1 column from the top-left endpoint
#      combo output = fold width 3 in boustrophedon rows:
#          [[2, 4, 7],
#           [7, 4, 2],
#           [2, 4, 7]]
#    The two changes interact: the color cycle determines both which endpoint is correct and how
#    many columns the folded output must have.
# b) Complexity calibration:
#    The generator uses 3-5 colors and chooses total path length as an exact multiple of that
#    count, with at least 4 folded rows. This keeps the repeated cycle obvious and gives several
#    row/column repetitions, so humans can see the fold width instead of guessing from a single
#    short chunk.
# c) Hidden-assumption audit of the source solver:
#    - `mask = g != bg` still assumes every non-bg cell belongs to the snake. Preserved.
#    - Endpoint search still assumes a simple chain with exactly two degree-1 cells. Preserved.
#    - `start = min(ends)` is no longer valid: traversal direction is now selected by matching the
#      repeating palette cycle, with a top-left fallback only for malformed inputs.
#    - `reshape((-1, 1))` is intentionally broken. The new output width is the palette size, and
#      the solver requires `len(colors) % width == 0`; the generator guarantees that divisibility.
# d) Placement legality:
#    `_try_extend` keeps the snake inside the border, forbids overlap, and rejects any new cell
#    that would touch an older snake cell other than its predecessor. That preserves one simple
#    non-self-touching path and keeps the traversal assumptions valid.
# e) Visual observability:
#    Across examples, the input snake visibly cycles through a small repeated palette, and the
#    output is not a column anymore: it is a folded rectangle whose column patterns depend on the
#    same cycle length. A human can infer the direction by following the repeating color rhythm,
#    then notice that the width equals the number of colors in that rhythm.
# f) Visual cleanliness:
#    There is only one object class in the input: one snake on a plain background. The output is
#    only the folded color matrix. No extra markers or overlays are added, so the visual field
#    stays uncluttered and the structural cue comes from repetition rather than decoration.
# g) Color-rule inferability:
#    The palette is capped at 3-5 colors and the stream is an exact ascending cycle, not an
#    arbitrary recoloring. That gives repeated support for the same rule in each sample and keeps
#    the color change tied to traversal position, which is observable from a few examples.
# 9) New solver assumptions and generator guarantees:
#    - Assumption: one simple snake path with exactly two endpoints. Guaranteed by path generation.
#    - Assumption: the intended direction is the traversal whose colors best match the ascending
#      palette cycle. Guaranteed by assigning cycle colors from the non-top-left endpoint.
#    - Assumption: folded width equals the number of distinct snake colors and divides the length.
#      Guaranteed by using every palette color and choosing total length as a multiple of it.
#    - Assumption: the combo should not collapse back to the source or to D1-only behavior.
#      Guaranteed by rejecting samples where top-left traversal yields the same folded matrix.

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
ARC_UTILS_DIR = ROOT / "arc2_solver_select"
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(ARC_UTILS_DIR))

from arc_utils import get_bg


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


def _fold_colors(colors: list[int], width: int) -> np.ndarray:
    if not colors:
        return np.zeros((0, 0), dtype=int)
    if width <= 0 or len(colors) % width != 0:
        raise ValueError("Color stream cannot be folded into a full matrix")

    height = len(colors) // width
    out = np.zeros((height, width), dtype=int)
    idx = 0
    for r in range(height):
        chunk = colors[idx : idx + width]
        idx += width
        out[r] = chunk if r % 2 == 0 else chunk[::-1]
    return out


def _extract_snake(g: np.ndarray, bg: int) -> tuple[list[tuple[int, int]], set[tuple[int, int]], list[tuple[int, int]]]:
    pts = [(int(r), int(c)) for r, c in np.argwhere(g != bg)]
    if not pts:
        return [], set(), []

    pset = set(pts)
    deg = {p: len(_path_neighbors(p, pset)) for p in pts}
    if any(d > 2 for d in deg.values()):
        raise ValueError("Expected a single chain without branches")

    ends = [p for p in pts if deg[p] == 1]
    if len(ends) != 2:
        raise ValueError("Expected exactly two snake endpoints")
    return pts, pset, ends


def _choose_traversal_colors(g: np.ndarray) -> tuple[list[int], list[int]]:
    bg = int(get_bg(g))
    pts, pset, ends = _extract_snake(g, bg)
    if not pts:
        return [], []

    palette = sorted({int(g[r, c]) for r, c in pts})
    traced_orders = [_trace_path(end, pset) for end in ends]
    traced_colors = [[int(g[r, c]) for r, c in order] for order in traced_orders]
    mismatch_scores = [_cycle_mismatch_count(colors, palette) for colors in traced_colors]

    if mismatch_scores[0] < mismatch_scores[1]:
        chosen = traced_colors[0]
    elif mismatch_scores[1] < mismatch_scores[0]:
        chosen = traced_colors[1]
    else:
        chosen = traced_colors[ends.index(min(ends))]
    return chosen, palette


def main(grid: np.ndarray) -> np.ndarray:
    """
    Choose the endpoint whose traversal follows the ascending repeating palette cycle,
    then fold that color stream into a boustrophedon matrix whose width is the cycle length.
    """
    g = np.array(grid, dtype=int)
    colors, palette = _choose_traversal_colors(g)
    if not colors:
        return np.zeros((0, 0), dtype=int)
    return _fold_colors(colors, len(palette))


def _try_extend(
    path_set: set[tuple[int, int]],
    last: tuple[int, int],
    dr: int,
    dc: int,
    count: int,
    H: int,
    W: int,
) -> list[tuple[int, int]]:
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


def _behavior_metrics(grid: np.ndarray) -> dict[str, int | bool]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    pts, pset, ends = _extract_snake(g, bg)
    palette = sorted({int(g[r, c]) for r, c in pts})
    width = len(palette)

    correct_colors, _ = _choose_traversal_colors(g)
    correct_out = _fold_colors(correct_colors, width)

    top_left = min(ends)
    top_left_order = _trace_path(top_left, pset)
    top_left_colors = [int(g[r, c]) for r, c in top_left_order]
    top_left_out = _fold_colors(top_left_colors, width)

    column_unique_counts = [len(set(correct_out[:, c].tolist())) for c in range(correct_out.shape[1])]
    return {
        "width": width,
        "height": int(correct_out.shape[0]),
        "top_left_diff_cells": int(np.sum(correct_out != top_left_out)),
        "cycle_mismatch_top_left": int(_cycle_mismatch_count(top_left_colors, palette)),
        "column_unique_max": max(column_unique_counts),
        "column_unique_min": min(column_unique_counts),
        "top_left_is_wrong": bool(not np.array_equal(correct_out, top_left_out)),
    }


def _gen_attempt() -> np.ndarray | None:
    H, W = 16, 16
    bg = random.randint(0, 9)
    color_count = random.randint(3, 5)
    palette = sorted(random.sample([c for c in range(10) if c != bg], color_count))

    row_choices = [rows for rows in range(4, 9) if rows * color_count <= 30]
    target_len = color_count * random.choice(row_choices)

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
    if total != target_len:
        return None
    if turns < 6 or turns > 12:
        return None

    endpoints = [path[0], path[-1]]
    top_left = min(endpoints)
    true_start = endpoints[0] if endpoints[1] == top_left else endpoints[1]
    oriented_path = path if path[0] == true_start else list(reversed(path))
    colors = _build_cyclic_colors(total, palette)

    grid = np.full((H, W), bg, dtype=int)
    for (r, c), col in zip(oriented_path, colors):
        grid[r, c] = col

    expected_out = _fold_colors(colors, color_count)
    actual_out = main(grid)
    if not np.array_equal(actual_out, expected_out):
        return None

    metrics = _behavior_metrics(grid)
    if not metrics["top_left_is_wrong"]:
        return None
    if metrics["column_unique_max"] > 2:
        return None
    if metrics["height"] < 4:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate input after 500 attempts")
