# Combo variant of 7b5033c1 (index 5)
# combo_spec: D1/0+D3/3
# Name: 多路径串联+折叠矩阵
#
# Design analysis
# a) Concrete behavior and trace:
#    Source task:
#    - All non-bg cells form one simple snake.
#    - Trace from the top-left endpoint.
#    - Emit one N x 1 color column.
#    Fusion task:
#    - Non-bg cells now form 3-5 disconnected short snakes, each still a simple
#      4-connected chain of length 5-8.
#    - Sort the snakes by their topmost/leftmost occupied cell, i.e. top-to-bottom
#      with left-to-right as tie-break.
#    - Trace each snake from its own top-left endpoint.
#    - Instead of concatenating into one column, place each traced color stream into
#      one output column. The output width M equals the number of snakes; the output
#      height N equals the longest snake length. Shorter columns are padded at the
#      bottom with the input background color.
#    Small example:
#      path A colors = [2,2,4,4,1]
#      path B colors = [3,3,6,6,6,8,8]
#      path C colors = [7,5,5,9,9,9]
#      sorted order = A, B, C
#      original task would be impossible because there is not one chain.
#      D3-only variant would output one long column:
#        [2,2,4,4,1,3,3,6,6,6,8,8,7,5,5,9,9,9]^T
#      fusion output is a 7 x 3 matrix:
#        [[2,3,7],
#         [2,3,5],
#         [4,6,5],
#         [4,6,9],
#         [1,6,9],
#         [bg,8,9],
#         [bg,8,bg]]
#    The key difference is that D3 changes how many source objects are read, while
#    D1 changes how the traced sequences are spatially arranged in the output.
# b) Complexity calibration:
#    - Number of snakes is limited to 3-5, so width inference has only a few values.
#    - Each snake length is limited to 5-8, so columns are short and visually legible.
#    - The generator forces at least two different snake lengths, making bottom padding
#      visible instead of collapsing to a full rectangle with no blank evidence.
#    - At least one row band contains two snakes, so the left-to-right tie-break is
#      actually exercised and not hidden behind a purely vertical ordering.
# c) Implicit-assumption audit of the original solver:
#    - `pts = np.argwhere(g != bg)` still means "all relevant cells", but the old code
#      assumed these points formed one component. That is false now, so we explicitly
#      split connected components first.
#    - Old degree counting on the whole set implicitly expected exactly two endpoints
#      globally. Now each component must have its own two endpoints, so tracing moves
#      into `_trace_component`.
#    - Old `start = min(ends)` still works, but only inside one component; applying it
#      globally would start on one snake and ignore the others.
#    - Old `reshape((-1, 1))` assumed output width is always 1. The fusion rule breaks
#      that and replaces it with `width = number of components`, `height = max length`.
# d) Placement legality:
#    - The generator uses small path templates that are valid simple chains by
#      construction.
#    - Components are placed in separated row bands; if two components share a band,
#      they also have a 3-5 cell horizontal gap.
#    - This prevents overlap, out-of-bounds placement, and accidental 4-neighbor
#      touching between snakes, so component extraction remains unambiguous.
# e) Visual observability:
#    - Humans can see several disconnected snakes in the input and several output
#      columns, so the multi-source and multi-column changes are both visible.
#    - Because one band always has two snakes, examples visibly need a left-to-right
#      tie-break, not just top-to-bottom.
#    - Because lengths differ, the bottom background padding reveals that columns are
#      per-snake traces rather than arbitrary grouped colors.
# f) Visual tidiness:
#    - Only snakes appear in the input; there are no extra markers or overlapping
#      annotations.
#    - Band placement keeps the canvas sparse but organized, so each snake is easy to
#      isolate by eye.
# g) Color inferability:
#    - No recoloring rule is introduced. Colors are preserved exactly from input trace
#      to output column.
#    - Palette size is kept small (3..num_paths), and each snake uses 2-4 contiguous
#      color segments, so color order is visible without becoming noisy.
# 9) New solver assumptions and generator guarantees:
#    - Assumption: background is the most frequent color. Guaranteed by using a 21x21
#      grid with only 15-40 non-bg cells.
#    - Assumption: every non-bg component is a simple chain with two endpoints.
#      Guaranteed by the path templates plus spacing.
#    - Assumption: lexicographic component order matches the intended top/down then
#      left/right reading order. Guaranteed by band placement and explicit same-band
#      left/right spacing.
#    - Assumption: output columns may have different lengths. Guaranteed by rejecting
#      samples where all path lengths are equal.

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/lixinyu/snap/new/solvers")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arc_utils import get_bg


NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
PATH_TEMPLATES: dict[int, list[list[tuple[int, int]]]] = {
    5: [
        [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2)],
        [(0, 0), (0, 1), (1, 1), (2, 1), (2, 2)],
    ],
    6: [
        [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (3, 2)],
        [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3)],
    ],
    7: [
        [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (3, 2), (3, 3)],
        [(0, 0), (0, 1), (1, 1), (2, 1), (2, 2), (3, 2), (3, 3)],
    ],
    8: [
        [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (3, 2), (3, 3), (4, 3)],
        [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3), (3, 3), (3, 4)],
    ],
}


def _connected_components(points: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    pset = set(points)
    seen: set[tuple[int, int]] = set()
    components: list[list[tuple[int, int]]] = []
    for start in sorted(points):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        comp: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            comp.append((r, c))
            for dr, dc in NEIGHBORS4:
                nxt = (r + dr, c + dc)
                if nxt in pset and nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        components.append(sorted(comp))
    return components


def _trace_component(component: list[tuple[int, int]]) -> list[tuple[int, int]]:
    pts = sorted(component)
    if not pts:
        return []

    pset = set(pts)
    deg: dict[tuple[int, int], int] = {}
    for r, c in pts:
        deg[(r, c)] = sum((r + dr, c + dc) in pset for dr, dc in NEIGHBORS4)

    assert all(d <= 2 for d in deg.values())
    if len(pts) == 1:
        return pts

    ends = [p for p in pts if deg[p] == 1]
    assert len(ends) == 2

    order: list[tuple[int, int]] = []
    prev: tuple[int, int] | None = None
    cur = min(ends)
    visited: set[tuple[int, int]] = set()
    while True:
        assert cur not in visited
        visited.add(cur)
        order.append(cur)
        r, c = cur
        nbrs = sorted(
            (r + dr, c + dc)
            for dr, dc in NEIGHBORS4
            if (r + dr, c + dc) in pset and (r + dr, c + dc) != prev
        )
        if not nbrs:
            break
        assert len(nbrs) == 1
        prev, cur = cur, nbrs[0]

    assert len(order) == len(pts)
    return order


def _pack_columns(color_streams: list[list[int]], bg: int) -> np.ndarray:
    if not color_streams:
        return np.zeros((0, 0), dtype=int)
    height = max(len(stream) for stream in color_streams)
    width = len(color_streams)
    out = np.full((height, width), bg, dtype=int)
    for c, stream in enumerate(color_streams):
        out[: len(stream), c] = np.array(stream, dtype=int)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fusion rule:
    - Find all disconnected non-background snakes.
    - Sort them by top-to-bottom, then left-to-right, using each component's minimum
      coordinate as its key.
    - Trace each snake from its own top-left endpoint.
    - Put each traced color sequence into one output column and pad shorter columns
      with background.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    points = [(int(r), int(c)) for r, c in np.argwhere(g != bg)]
    if not points:
        return np.zeros((0, 0), dtype=int)

    components = _connected_components(points)
    components.sort(key=min)

    color_streams: list[list[int]] = []
    for comp in components:
        order = _trace_component(comp)
        color_streams.append([int(g[r, c]) for r, c in order])

    return _pack_columns(color_streams, bg)


def _random_partition(total: int, num_parts: int, min_len: int, max_len: int) -> list[int] | None:
    if num_parts * min_len > total or num_parts * max_len < total:
        return None
    parts = [min_len] * num_parts
    remaining = total - sum(parts)
    for _ in range(remaining):
        eligible = [i for i, value in enumerate(parts) if value < max_len]
        if not eligible:
            return None
        parts[random.choice(eligible)] += 1
    random.shuffle(parts)
    return parts


def _transform_path(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    arr = np.array(path, dtype=int)
    if random.random() < 0.5:
        arr[:, 0] *= -1
    if random.random() < 0.5:
        arr[:, 1] *= -1
    if random.random() < 0.5:
        arr = arr[:, ::-1]
    arr[:, 0] -= int(arr[:, 0].min())
    arr[:, 1] -= int(arr[:, 1].min())
    return [(int(r), int(c)) for r, c in arr.tolist()]


def _sample_relative_path(length: int) -> list[tuple[int, int]]:
    return _transform_path(random.choice(PATH_TEMPLATES[length]))


def _ensure_required_color(seg_colors: list[int], required: int | None) -> list[int]:
    if required is None or required in seg_colors:
        return seg_colors
    for idx in range(len(seg_colors) - 1, -1, -1):
        prev_col = seg_colors[idx - 1] if idx > 0 else None
        next_col = seg_colors[idx + 1] if idx + 1 < len(seg_colors) else None
        if required != prev_col and required != next_col:
            seg_colors[idx] = required
            return seg_colors
    seg_colors[0] = required
    return seg_colors


def _make_path_colors(
    length: int,
    palette: list[int],
    required: int | None,
    avoid_first: int | None,
) -> list[int]:
    num_segments = random.randint(2, min(4, length))
    seg_lengths = _random_partition(length, num_segments, min_len=1, max_len=4)
    assert seg_lengths is not None

    first_choices = [c for c in palette if c != avoid_first] or list(palette)
    seg_colors = [random.choice(first_choices)]
    for _ in range(1, num_segments):
        choices = [c for c in palette if c != seg_colors[-1]]
        seg_colors.append(random.choice(choices))

    seg_colors = _ensure_required_color(seg_colors, required)
    for i in range(1, len(seg_colors)):
        if seg_colors[i] == seg_colors[i - 1]:
            choices = [c for c in palette if c != seg_colors[i - 1]]
            seg_colors[i] = random.choice(choices)

    colors: list[int] = []
    for seg_len, color in zip(seg_lengths, seg_colors):
        colors.extend([color] * seg_len)
    return colors


def _plan_band_counts(num_paths: int) -> tuple[list[int], list[int]]:
    patterns = {
        3: [[2, 1], [1, 2]],
        4: [[2, 2], [2, 1, 1], [1, 2, 1], [1, 1, 2]],
        5: [[2, 2, 1], [2, 1, 2], [1, 2, 2]],
    }
    counts = random.choice(patterns[num_paths])
    rows = sorted(random.sample([1, 8, 15], len(counts)))
    return rows, counts


def _place_band_components(
    band_row: int,
    shapes: list[list[tuple[int, int]]],
    H: int,
    W: int,
) -> list[list[tuple[int, int]]]:
    dims: list[tuple[list[tuple[int, int]], int]] = []
    for shape in shapes:
        height = max(r for r, _ in shape) + 1
        width = max(c for _, c in shape) + 1
        assert band_row + height <= H - 1
        dims.append((shape, width))

    if len(dims) == 1:
        shape, width = dims[0]
        col = random.randint(1, W - width - 2)
        return [[(band_row + r, col + c) for r, c in shape]]

    assert len(dims) == 2
    gap = random.randint(3, 5)
    total_width = dims[0][1] + dims[1][1] + gap
    start_col = random.randint(1, W - total_width - 2)
    placed: list[list[tuple[int, int]]] = []
    col = start_col
    for shape, width in dims:
        placed.append([(band_row + r, col + c) for r, c in shape])
        col += width + gap
    return placed


def _gen_attempt() -> np.ndarray | None:
    H, W = 21, 21
    bg = random.randint(0, 9)
    num_paths = random.randint(3, 5)
    palette_size = random.randint(3, num_paths)
    palette = random.sample([c for c in range(10) if c != bg], palette_size)

    lengths = [random.randint(5, 8) for _ in range(num_paths)]
    if len(set(lengths)) == 1:
        return None

    band_rows, band_counts = _plan_band_counts(num_paths)
    random.shuffle(lengths)

    components: list[list[tuple[int, int]]] = []
    idx = 0
    for band_row, band_count in zip(band_rows, band_counts):
        shapes = [_sample_relative_path(lengths[idx + j]) for j in range(band_count)]
        idx += band_count
        components.extend(_place_band_components(band_row, shapes, H, W))

    all_points = [pt for comp in components for pt in comp]
    if len(all_points) != len(set(all_points)):
        return None

    traced_components = [_trace_component(comp) for comp in components]
    traced_components.sort(key=min)
    traced_lengths = [len(comp) for comp in traced_components]
    if len(set(traced_lengths)) == 1:
        return None

    grid = np.full((H, W), bg, dtype=int)
    expected_streams: list[list[int]] = []
    needed = list(palette)
    random.shuffle(needed)
    prev_first: int | None = None
    for order in traced_components:
        required = needed.pop() if needed else None
        colors = _make_path_colors(len(order), palette, required=required, avoid_first=prev_first)
        prev_first = colors[0]
        expected_streams.append(colors)
        for (r, c), color in zip(order, colors):
            grid[r, c] = color

    comps_from_grid = _connected_components([(int(r), int(c)) for r, c in np.argwhere(grid != bg)])
    if len(comps_from_grid) != num_paths:
        return None
    if any(not (5 <= len(comp) <= 8) for comp in comps_from_grid):
        return None

    expected_out = _pack_columns(expected_streams, bg)
    actual_out = main(grid)
    if not np.array_equal(actual_out, expected_out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("Failed to generate input after 500 attempts")
