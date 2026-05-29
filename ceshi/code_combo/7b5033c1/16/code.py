# Combo variant of 7b5033c1 (index 16)
# combo_spec: D9/2+D10/1
# Name: 条件过滤+颜色映射表

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


# Design analysis for the fused variant:
# a) Concrete behavior and trace:
#    Original 7b5033c1 input contains only one multicolor snake. The solver traces the unique path
#    from the top-left-most endpoint and outputs every visited color as an Nx1 column.
#    This combo adds a second disconnected component: a 2xK legend strip on the top or bottom edge.
#    In each legend column, the upper cell is a source color and the lower cell is its mapped output
#    color. The snake is still traced exactly as before, but the 1D color sequence is then split into
#    maximal same-color runs; only runs of length >= 3 survive, and every surviving pixel is recolored
#    through the legend before being written to the output column.
#    Example:
#    - traced snake colors: [2,2,2, 5, 7,7, 2,2,2,2]
#    - legend pairs: 2->8, 5->1, 7->3
#    - original task output: [2,2,2,5,7,7,2,2,2,2]^T
#    - fused output: keep only runs len>=3 -> [2,2,2,2,2,2,2], then recolor -> [8,8,8,8,8,8,8]^T
#    The differences happen in two coordinated stages: deletion depends on run length, and the remaining
#    colors are not copied directly but symbolically decoded through the edge legend.
# b) Complexity calibration:
#    The filter uses one discrete threshold only: short runs (1-2) vanish, long runs (>=3) survive.
#    The legend uses only 3-4 columns, so the color mapping space stays small enough to infer from a
#    few examples. The generator forces both short and long runs, and also forces at least one color
#    to appear in both a short run and a long run, so the task does not collapse into "keep some fixed
#    colors" or into the original unfiltered straightening.
# c) Implicit-assumption audit against the original main():
#    - `mask = g != bg` no longer means "all non-bg cells belong to one snake"; the legend is also
#      non-background, so component separation is now mandatory.
#    - Degree counting / endpoint detection only makes sense on the snake component. Running them on the
#      full non-bg mask would mix in the legend rectangle and break the chain assumptions.
#    - The original direct `return all traced colors` is no longer valid. The new output stage assumes:
#      1) the traced color sequence can be partitioned into adjacent-equal runs,
#      2) every surviving color appears in the legend's source row,
#      3) the legend columns define a one-to-one recoloring map.
#      The generator explicitly constructs samples that satisfy all three.
# d) Placement legality:
#    The legend occupies a full 2xK rectangle on the top two rows or bottom two rows, with K in {3,4}.
#    The snake is generated inside an interior corridor that leaves at least two background rows between
#    it and the legend, so the two components never touch, overlap, or create accidental bridges.
#    Path growth still checks bounds, self-overlap, and forbidden side-adjacencies, preserving a simple
#    4-connected chain with exactly two endpoints.
# e) Visual observability:
#    Across examples, the output remains a single column following the snake's traversal order, so the
#    original task is still recognizable. What visibly changes is that tiny 1-2-cell color blips vanish
#    while the remaining colors are replaced by the legend's second row. A human can compare the snake,
#    the legend columns, and the shorter recolored output to infer both operations.
# f) Visual cleanliness:
#    The instruction channel is isolated on one edge as a compact rectangle; the snake remains in the
#    interior. There is no overlap between legend and snake, no border noise, and no third object class.
#    This keeps the grid readable instead of mixing symbolic hints into the same area as the path.
# g) Color-change inferability:
#    The mapping is not hidden: every recolored output value is explicitly paired with its source color
#    in the 2xK legend. The source palette is limited to 3-4 colors, the target palette is also small,
#    and the generator uses every source color at least once. Because one source color appears in both
#    kept and deleted runs, the learner can distinguish "mapped because kept" from "deleted despite being
#    listed in the legend".
# Post-modification solver assumption audit:
#    - Exactly one connected component is a full 2xK border rectangle; that is the legend.
#    - After removing that legend, exactly one connected component remains and it is a simple chain.
#    - The upper legend row contains distinct source colors, and the lower row contains their targets.
#    - At least one run is deleted and at least two long runs survive, so the combo behavior is active.
#    `generate_input()` enforces these assumptions by construction and by self-verifying with `main()`.


Coord = tuple[int, int]


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
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
            cr, cc2 = stack.pop()
            comp.append((cr, cc2))
            for nr, nc in _neighbors4(cr, cc2):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(comp)

    return comps


def _legend_mapping_if_valid(grid: np.ndarray, comp: list[Coord]) -> dict[int, int] | None:
    h, w = grid.shape
    rows = sorted({r for r, _ in comp})
    cols = sorted({c for _, c in comp})
    if len(rows) != 2:
        return None
    if rows[1] != rows[0] + 1:
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

    mapping: dict[int, int] = {}
    for c in cols:
        src = int(grid[rows[0], c])
        dst = int(grid[rows[1], c])
        if src in mapping:
            return None
        mapping[src] = dst
    return mapping


def _decode_legend_and_path(grid: np.ndarray, bg: int) -> tuple[dict[int, int], list[Coord]]:
    mask = grid != bg
    comps = _find_components(mask)
    if len(comps) != 2:
        raise ValueError(f"expected exactly two non-bg components, found {len(comps)}")

    legend_mapping: dict[int, int] | None = None
    path_comp: list[Coord] | None = None
    for comp in comps:
        maybe_mapping = _legend_mapping_if_valid(grid, comp)
        if maybe_mapping is not None:
            if legend_mapping is not None:
                raise ValueError("found multiple legend-like components")
            legend_mapping = maybe_mapping
        else:
            if path_comp is not None:
                raise ValueError("found multiple non-legend components")
            path_comp = comp

    if legend_mapping is None or path_comp is None:
        raise ValueError("failed to separate legend and snake")

    return legend_mapping, path_comp


def _trace_path(pts: list[Coord]) -> list[Coord]:
    pset = set(pts)
    deg: dict[Coord, int] = {}
    for r, c in pts:
        deg[(r, c)] = sum((nr, nc) in pset for nr, nc in _neighbors4(r, c))

    ends = [p for p in pts if deg[p] == 1]
    if len(ends) != 2:
        raise ValueError(f"expected exactly two snake endpoints, found {len(ends)}")
    start = min(ends)

    order: list[Coord] = []
    prev: Coord | None = None
    cur = start
    for _ in range(len(pts)):
        order.append(cur)
        nbrs = [q for q in _neighbors4(*cur) if q in pset and q != prev]
        if not nbrs:
            break
        if len(nbrs) != 1:
            raise ValueError("snake is not a simple chain")
        prev, cur = cur, nbrs[0]

    if len(order) != len(pts):
        raise ValueError("failed to traverse every snake cell")
    return order


def _filter_runs(colors: list[int], min_len: int = 3) -> list[int]:
    kept: list[int] = []
    i = 0
    while i < len(colors):
        j = i + 1
        while j < len(colors) and colors[j] == colors[i]:
            j += 1
        if j - i >= min_len:
            kept.extend(colors[i:j])
        i = j
    return kept


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace the snake, delete short color runs, then recolor the survivors through a 2xK legend.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    mapping, pts = _decode_legend_and_path(g, bg)
    if not pts:
        return np.zeros((0, 1), dtype=int)

    order = _trace_path(pts)
    traced = [int(g[r, c]) for r, c in order]
    filtered = _filter_runs(traced, min_len=3)
    recolored = [int(mapping[col]) for col in filtered]
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
        for nb in _neighbors4(nr, nc):
            if nb != pred and nb in path_set:
                bad = True
                break
        if bad:
            break

        segment.append((nr, nc))
        path_set.add((nr, nc))
    return segment


def _random_typed_partition(total: int) -> list[int] | None:
    for _ in range(400):
        num_segments = random.randint(6, 8)
        short_count = random.randint(2, min(4, num_segments - 2))
        short_idx = set(random.sample(range(num_segments), short_count))

        parts = [1 if i in short_idx else 3 for i in range(num_segments)]
        caps = [2 if i in short_idx else 8 for i in range(num_segments)]
        remaining = total - sum(parts)
        if remaining < 0:
            continue

        while remaining > 0:
            eligible = [i for i in range(num_segments) if parts[i] < caps[i]]
            if not eligible:
                break
            pick = random.choice(eligible)
            parts[pick] += 1
            remaining -= 1
        if remaining != 0:
            continue

        short_pixels = sum(parts[i] for i in short_idx)
        long_runs = [parts[i] for i in range(num_segments) if i not in short_idx]
        if short_pixels < 4:
            continue
        if len(long_runs) < 2 or sum(long_runs) < 8:
            continue
        return parts

    return None


def _build_segment_colors(seg_lengths: list[int], palette: list[int]) -> list[int] | None:
    long_idx = [i for i, x in enumerate(seg_lengths) if x >= 3]
    short_idx = [i for i, x in enumerate(seg_lengths) if x < 3]
    for _ in range(500):
        seg_colors: list[int] = []
        for _i in range(len(seg_lengths)):
            choices = [c for c in palette if not seg_colors or c != seg_colors[-1]]
            seg_colors.append(int(random.choice(choices)))

        if len(set(seg_colors)) != len(palette):
            continue
        long_colors = {seg_colors[i] for i in long_idx}
        if len(long_colors) < 2:
            continue

        mixed_color_exists = False
        for col in palette:
            has_long = any(seg_colors[i] == col and seg_lengths[i] >= 3 for i in range(len(seg_lengths)))
            has_short = any(seg_colors[i] == col and seg_lengths[i] < 3 for i in range(len(seg_lengths)))
            if has_long and has_short:
                mixed_color_exists = True
                break
        if not mixed_color_exists:
            continue

        return seg_colors

    return None


def _build_mapping(source_palette: list[int], usable: list[int]) -> dict[int, int] | None:
    remaining = [c for c in usable if c not in source_palette]
    if len(remaining) < len(source_palette):
        return None
    targets = random.sample(remaining, len(source_palette))
    random.shuffle(targets)
    return {int(src): int(dst) for src, dst in zip(source_palette, targets)}


def _paint_legend(
    grid: np.ndarray,
    mapping: dict[int, int],
    side: str,
) -> None:
    cols = list(mapping.keys())
    random.shuffle(cols)
    k = len(cols)
    start_c = random.randint(1, grid.shape[1] - k - 1)
    if side == "top":
        top_r = 0
    else:
        top_r = grid.shape[0] - 2

    for offset, src in enumerate(cols):
        c = start_c + offset
        grid[top_r, c] = int(src)
        grid[top_r + 1, c] = int(mapping[src])


def _expected_output(path: list[Coord], colors: list[int], mapping: dict[int, int]) -> np.ndarray:
    start = min(path[0], path[-1])
    oriented = list(colors) if path[0] == start else list(reversed(colors))
    filtered = _filter_runs(oriented, min_len=3)
    recolored = [int(mapping[col]) for col in filtered]
    return np.array(recolored, dtype=int).reshape((-1, 1))


def _gen_attempt() -> np.ndarray | None:
    h, w = 16, 16
    bg = random.randint(0, 9)
    usable = [c for c in range(10) if c != bg]

    legend_side = random.choice(["top", "bottom"])
    source_count = random.randint(3, 4)
    source_palette = random.sample(usable, source_count)
    mapping = _build_mapping(source_palette, usable)
    if mapping is None:
        return None

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
        segment: list[Coord] = []

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

    seg_lengths = _random_typed_partition(total)
    if seg_lengths is None:
        return None
    seg_colors = _build_segment_colors(seg_lengths, source_palette)
    if seg_colors is None:
        return None

    colors: list[int] = []
    for seg_len, col in zip(seg_lengths, seg_colors):
        colors.extend([int(col)] * int(seg_len))

    grid = np.full((h, w), bg, dtype=int)
    for (r, c), col in zip(path, colors):
        grid[r, c] = int(col)
    _paint_legend(grid, mapping, legend_side)

    expected = _expected_output(path, colors, mapping)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    filtered_src = _filter_runs(list(colors) if path[0] == min(path[0], path[-1]) else list(reversed(colors)))
    if not filtered_src:
        return None
    if actual.shape[0] >= total:
        return None
    if actual.shape[0] < 8:
        return None
    if any(int(mapping[c]) == int(c) for c in filtered_src):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate input after 500 attempts")
