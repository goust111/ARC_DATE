# Combo variant of 7b5033c1 (index 21)
# combo_spec: D4/1+D10/1
# Name: 渐变颜色+颜色映射表

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


# Design analysis for the fused variant
# a) Concrete behavior and trace:
#    Original 7b5033c1 has one non-background snake, starts from the top-left endpoint, and emits
#    the raw traced colors as an Nx1 column.
#    This combo adds a disconnected 2xK legend on the top or bottom edge. The legend has two aligned
#    roles:
#    - its top row is the intended repeating traversal cycle order, e.g. [6,2,9]
#    - its bottom row is the recoloring map for those same source colors, e.g. [1,7,4]
#    A concrete sample can therefore behave like this:
#    - legend top/bottom:  [6,2,9]
#                         [1,7,4]
#    - one snake endpoint gives source colors [6,2,9,6,2,9,6]
#    - the opposite endpoint gives          [6,9,2,6,9,2,6] only if geometry is read backward
#      or more commonly the exact reverse sequence [6,9,2,6,9,2,6] for a length-7 example
#    - original solver would choose the top-left endpoint and output the raw source colors
#    - fused solver traces from both endpoints, keeps the traversal that best matches the legend's
#      repeated top row, then remaps each color through the bottom row and outputs [1,7,4,1,7,4,1]^T
#    The two dimensions cooperate: the legend is not a decorative second object. Its column order
#    determines the traversal direction, and those same columns define the recoloring.
# b) Complexity calibration:
#    The generator uses 3-4 legend columns and snake lengths 20-30, so each cycle repeats many
#    times and the direction cue is strong. Keeping K small makes the mapping inferable from 2-4
#    training pairs, while repeated cycles prevent accidental ambiguity.
# c) Implicit-assumption audit against the original `main()`:
#    - `g != bg` no longer isolates a single snake; it now includes the legend rectangle too, so
#      component separation is required before any degree counting.
#    - `start = min(endpoints)` is no longer valid. Direction must be inferred from the legend cycle.
#    - `cols = [g[r,c] for ...]` is no longer the final answer. The traced colors are intermediate
#      symbolic tokens that must be recolored through the legend.
#    - The new solver assumes exactly one legend-like 2xK border rectangle and exactly one remaining
#      snake component. `generate_input()` constructs and verifies that.
# d) Placement legality:
#    The legend is a full 2xK rectangle flush with the top or bottom edge. The snake is generated in
#    an interior band that leaves at least two blank rows between it and the legend, so the two
#    components never touch, overlap, or create accidental bridges. Path growth still forbids revisits
#    and forbidden side-adjacencies, preserving a simple chain.
# e) Visual observability:
#    A human can compare the output strip to the legend bottom row and decode it back to legend-top
#    source colors. Those decoded colors visibly repeat in the snake from only one endpoint in the
#    legend's left-to-right order. That makes both "choose direction" and "then remap colors"
#    observable from examples.
# f) Visual cleanliness:
#    There are only two object classes: one interior snake and one compact edge legend. No markers are
#    overlaid onto the snake itself, and the legend stays rectangular and isolated on a border, so the
#    scene stays easy to parse.
# g) Color-change inferability:
#    The recoloring is explicit, not hidden. Every output color is paired with its source color in the
#    legend. The generator uses every source color multiple times along the snake and chooses target
#    colors disjoint from source colors, so the remapping remains visually obvious instead of collapsing
#    into identity or near-identity behavior.
# h) Quantitative anti-degeneracy metric:
#    `variant_metrics()` reports:
#    - `direction_margin`: mismatch gap between the two endpoint traversals against the legend cycle
#    - `top_left_is_wrong_start`: whether the original top-left rule would choose the wrong endpoint
#    - `mapped_output_differs_from_original`: whether fused output differs from the original task's raw
#      top-left output on the same grid
#    - `mapping_changes_all_pixels`: whether every output pixel color differs from its source color
#    The generator rejects samples unless all of these are active, which prevents regression to the
#    seed behavior.
# i) New assumptions introduced by the fused `main()`:
#    - The legend top row order is the cycle head order; reversing the legend would change the answer.
#    - All snake colors belong to the legend source palette.
#    - The snake traversal that follows the repeated legend-top cycle is unique.
#    `generate_input()` guarantees these conditions by construction and by self-checking with
#    `variant_metrics()`.

Coord = tuple[int, int]
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _neighbors4(cell: Coord) -> tuple[Coord, Coord, Coord, Coord]:
    r, c = cell
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_components(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[Coord]] = []

    for rr, cc in np.argwhere(mask):
        r = int(rr)
        c = int(cc)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
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


def _legend_info_if_valid(grid: np.ndarray, comp: list[Coord]) -> tuple[list[int], dict[int, int]] | None:
    h, w = grid.shape
    rows = sorted({r for r, _ in comp})
    cols = sorted({c for _, c in comp})
    if len(rows) != 2 or rows[1] != rows[0] + 1:
        return None
    if rows not in ([0, 1], [h - 2, h - 1]):
        return None
    if len(cols) < 3 or len(cols) > 4:
        return None
    if cols != list(range(cols[0], cols[0] + len(cols))):
        return None
    if len(comp) != 2 * len(cols):
        return None

    cells = set(comp)
    for r in rows:
        for c in cols:
            if (r, c) not in cells:
                return None

    source_order: list[int] = []
    mapping: dict[int, int] = {}
    targets_seen: set[int] = set()
    for c in cols:
        src = int(grid[rows[0], c])
        dst = int(grid[rows[1], c])
        if src in mapping or dst in targets_seen:
            return None
        source_order.append(src)
        mapping[src] = dst
        targets_seen.add(dst)
    return source_order, mapping


def _decode_legend_and_path(grid: np.ndarray, bg: int) -> tuple[list[int], dict[int, int], list[Coord]]:
    mask = grid != bg
    comps = _find_components(mask)
    if len(comps) != 2:
        raise ValueError(f"expected exactly two non-background components, found {len(comps)}")

    legend_order: list[int] | None = None
    legend_mapping: dict[int, int] | None = None
    path_comp: list[Coord] | None = None
    for comp in comps:
        info = _legend_info_if_valid(grid, comp)
        if info is not None:
            if legend_order is not None:
                raise ValueError("found multiple legend-like components")
            legend_order, legend_mapping = info
        else:
            if path_comp is not None:
                raise ValueError("found multiple non-legend components")
            path_comp = comp

    if legend_order is None or legend_mapping is None or path_comp is None:
        raise ValueError("failed to separate legend and snake")
    return legend_order, legend_mapping, path_comp


def _path_neighbors(cell: Coord, pset: set[Coord]) -> list[Coord]:
    return [nb for nb in _neighbors4(cell) if nb in pset]


def _trace_path(start: Coord, pset: set[Coord]) -> list[Coord]:
    order: list[Coord] = []
    prev: Coord | None = None
    cur = start

    for _ in range(len(pset)):
        order.append(cur)
        nxts = [q for q in _path_neighbors(cur, pset) if q != prev]
        if not nxts:
            break
        if len(nxts) != 1:
            raise ValueError("path is not a simple chain")
        prev, cur = cur, nxts[0]

    if len(order) != len(pset):
        raise ValueError("failed to traverse the full path")
    return order


def _cycle_mismatch_count(colors: list[int], cycle: list[int]) -> int:
    period = len(cycle)
    return sum(int(col != cycle[i % period]) for i, col in enumerate(colors))


def _choose_direction(
    grid: np.ndarray,
    pts: list[Coord],
    cycle: list[int],
) -> tuple[list[int], dict[str, int]]:
    pset = set(pts)
    deg = {p: len(_path_neighbors(p, pset)) for p in pts}
    ends = [p for p in pts if deg[p] == 1]
    if len(ends) != 2:
        raise ValueError(f"expected exactly two snake endpoints, found {len(ends)}")

    traced_orders = [_trace_path(end, pset) for end in ends]
    traced_colors = [[int(grid[r, c]) for r, c in order] for order in traced_orders]
    scores = [_cycle_mismatch_count(colors, cycle) for colors in traced_colors]

    if scores[0] < scores[1]:
        chosen_idx = 0
    elif scores[1] < scores[0]:
        chosen_idx = 1
    else:
        chosen_idx = ends.index(min(ends))

    return traced_colors[chosen_idx], {
        "direction_margin": abs(scores[0] - scores[1]),
        "chosen_index": chosen_idx,
        "top_left_index": ends.index(min(ends)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the border legend, choose the snake direction that follows the repeated legend-top cycle,
    then remap the linearized colors through the legend-bottom row.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    cycle, mapping, pts = _decode_legend_and_path(g, bg)
    if not pts:
        return np.zeros((0, 1), dtype=int)

    chosen_colors, _meta = _choose_direction(g, pts, cycle)
    recolored = [int(mapping[col]) for col in chosen_colors]
    return np.array(recolored, dtype=int).reshape((-1, 1))


def _try_extend(
    path_set: set[Coord],
    last: Coord,
    dr: int,
    dc: int,
    count: int,
    row_min: int,
    row_max: int,
    col_min: int,
    col_max: int,
) -> list[Coord]:
    segment: list[Coord] = []
    r, c = last
    for i in range(1, count + 1):
        nr = r + dr * i
        nc = c + dc * i
        if nr < row_min or nr > row_max or nc < col_min or nc > col_max:
            break
        if (nr, nc) in path_set:
            break

        pred = (r + dr * (i - 1), c + dc * (i - 1))
        bad = False
        for nb in _neighbors4((nr, nc)):
            if nb != pred and nb in path_set:
                bad = True
                break
        if bad:
            break

        segment.append((nr, nc))
        path_set.add((nr, nc))
    return segment


def _build_mapping(source_order: list[int], usable: list[int]) -> dict[int, int] | None:
    remaining = [c for c in usable if c not in source_order]
    if len(remaining) < len(source_order):
        return None
    targets = random.sample(remaining, len(source_order))
    return {int(src): int(dst) for src, dst in zip(source_order, targets)}


def _paint_legend(grid: np.ndarray, source_order: list[int], mapping: dict[int, int], side: str) -> None:
    k = len(source_order)
    start_c = random.randint(1, grid.shape[1] - k - 1)
    top_r = 0 if side == "top" else grid.shape[0] - 2
    for offset, src in enumerate(source_order):
        c = start_c + offset
        grid[top_r, c] = int(src)
        grid[top_r + 1, c] = int(mapping[src])


def _expected_output(path: list[Coord], colors: list[int], mapping: dict[int, int]) -> np.ndarray:
    recolored = [int(mapping[col]) for col in colors]
    return np.array(recolored, dtype=int).reshape((-1, 1))


def variant_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    cycle, mapping, pts = _decode_legend_and_path(g, bg)
    chosen_colors, meta = _choose_direction(g, pts, cycle)

    pset = set(pts)
    deg = {p: len(_path_neighbors(p, pset)) for p in pts}
    ends = [p for p in pts if deg[p] == 1]
    top_left = min(ends)
    top_left_order = _trace_path(top_left, pset)
    top_left_colors = [int(g[r, c]) for r, c in top_left_order]

    mapped_variant = [int(mapping[col]) for col in chosen_colors]
    mapped_top_left = [int(mapping[col]) for col in top_left_colors]
    original_raw = top_left_colors

    return {
        "legend_size": len(cycle),
        "path_length": len(chosen_colors),
        "direction_margin": int(meta["direction_margin"]),
        "top_left_is_wrong_start": int(meta["chosen_index"] != meta["top_left_index"]),
        "mapped_output_differs_from_original": int(mapped_variant != original_raw),
        "direction_changes_mapped_output": int(mapped_variant != mapped_top_left),
        "mapping_changes_all_pixels": int(all(src != dst for src, dst in zip(chosen_colors, mapped_variant))),
    }


def _gen_attempt() -> np.ndarray | None:
    h, w = 16, 16
    bg = random.randint(0, 9)
    usable = [c for c in range(10) if c != bg]

    source_count = random.randint(3, 4)
    source_order = random.sample(usable, source_count)
    mapping = _build_mapping(source_order, usable)
    if mapping is None:
        return None

    legend_side = random.choice(["top", "bottom"])
    if legend_side == "top":
        row_min, row_max = 4, h - 2
    else:
        row_min, row_max = 1, h - 5
    col_min, col_max = 1, w - 2

    target_len = random.randint(20, 30)
    start_r = random.randint(row_min, min(row_min + 3, row_max))
    start_c = random.randint(col_min, col_max)

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
            segment = _try_extend(path_set, last, v_dir, 0, run_len, row_min, row_max, col_min, col_max)
            if not segment:
                segment = _try_extend(path_set, last, -v_dir, 0, run_len, row_min, row_max, col_min, col_max)
        else:
            segment = _try_extend(path_set, last, 0, h_dir, run_len, row_min, row_max, col_min, col_max)
            if not segment:
                segment = _try_extend(path_set, last, 0, -h_dir, run_len, row_min, row_max, col_min, col_max)
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
    colors = [int(source_order[i % len(source_order)]) for i in range(total)]

    grid = np.full((h, w), bg, dtype=int)
    for (r, c), col in zip(oriented_path, colors):
        grid[r, c] = int(col)
    _paint_legend(grid, source_order, mapping, legend_side)

    expected = _expected_output(oriented_path, colors, mapping)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    metrics = variant_metrics(grid)
    if metrics["direction_margin"] <= 0:
        return None
    if metrics["top_left_is_wrong_start"] != 1:
        return None
    if metrics["direction_changes_mapped_output"] != 1:
        return None
    if metrics["mapped_output_differs_from_original"] != 1:
        return None
    if metrics["mapping_changes_all_pixels"] != 1:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate input after 500 attempts")
