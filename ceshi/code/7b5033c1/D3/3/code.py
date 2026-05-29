# Variant of 7b5033c1
# Dimension: D3 — 源对象：蛇形路径结构 (src)
# Change: 单条4-连通无分支蛇形路径，20-30像素，阶梯形交替水平/垂直走向，7-11个转弯... → 多条独立短路径

from __future__ import annotations

from collections import Counter
import random

import numpy as np


# Design analysis for this D3 variant
# a) Concrete behavior:
#    The original task has one long snake, so the solver can treat all non-bg cells
#    as a single connected chain. The variant instead places 3-5 disconnected short
#    snakes. Each component is still a simple 4-connected path, but the output is now
#    the concatenation of several traced components. We make the inter-component order
#    explicit and visible: sort components by their topmost/leftmost cell (top-to-bottom,
#    left-to-right), then trace each component from its own top-left-most endpoint.
#    Small example:
#    - Original: one chain A-B-C-D => output [A,B,C,D].
#    - Variant: two chains (A-B-C) above (D-E-F) => output [A,B,C,D,E,F].
#    The changed step is therefore "which source object(s) feed the output", not the
#    within-object tracing rule.
# b) Complexity calibration:
#    Keeping each component as a short snake preserves the original local reasoning,
#    but having 3-5 components introduces a real object-sorting problem. The components
#    are separated into row bands, and some samples contain two snakes in the same band,
#    so the left-to-right tiebreak is visually observable instead of collapsing into a
#    trivial always-top-to-bottom pattern.
# c) Old implicit assumptions that break:
#    - `pts` was assumed to belong to one component. Not true now.
#    - `ends = [p for p in pts if deg[p] == 1]` previously found the two endpoints of
#      the whole object. Now every component has its own two endpoints, so degree must
#      be computed and interpreted per component.
#    - The single traversal loop stopped after one chain. In the variant that would
#      ignore the remaining snakes.
# d) Placement legality:
#    The generator places components in 3 separated row bands, with at most 2 snakes
#    per band and at least a 2-column gap inside a band. This guarantees no overlap,
#    no 4-neighbor touching between components, and keeps every component a valid chain
#    that the solver can trace independently.
# e) Visual observability:
#    From examples, a human can see that the input contains several disconnected snakes
#    while the output is one long column. Because output length exceeds any one local
#    snake and some samples require same-row left-to-right ordering, the variant is
#    distinguishable from the original single-snake task.
# New solver assumptions and generator guarantees:
# - Every non-bg connected component is a simple chain: generator enforces this using
#   path templates and spacing; `_trace_component` asserts it.
# - Component order is lexicographic on each component's minimum cell: generator places
#   components so this produces the intended top-to-bottom, left-to-right order.
# - Each component length is 5-8 and the number of components is 3-5, so the variant
#   cannot degenerate back into the original single-chain case.


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


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.array(grid, dtype=int).flatten())).most_common(1)[0][0])


def _connected_components(points: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    pset = set(points)
    components: list[list[tuple[int, int]]] = []
    seen: set[tuple[int, int]] = set()
    for start in sorted(points):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        comp: list[tuple[int, int]] = []
        while stack:
            cur = stack.pop()
            comp.append(cur)
            r, c = cur
            for dr, dc in NEIGHBORS4:
                nxt = (r + dr, c + dc)
                if nxt in pset and nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        components.append(sorted(comp))
    return components


def _trace_component(component: list[tuple[int, int]]) -> list[tuple[int, int]]:
    pts = sorted(component)
    if len(pts) <= 1:
        return pts

    pset = set(pts)
    deg: dict[tuple[int, int], int] = {}
    for r, c in pts:
        deg[(r, c)] = sum((r + dr, c + dc) in pset for dr, dc in NEIGHBORS4)

    ends = [p for p in pts if deg[p] == 1]
    mids = [p for p in pts if deg[p] == 2]
    assert len(ends) == 2
    assert len(mids) == len(pts) - 2

    order: list[tuple[int, int]] = []
    start = min(ends)
    prev: tuple[int, int] | None = None
    cur = start
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule:
    - Non-bg cells form 3-5 disconnected simple snakes.
    - Sort the snake components by their topmost/leftmost occupied cell.
    - For each snake, start at its top-left-most endpoint and trace along the chain.
    - Concatenate all traced colors into one output column.
    """
    g = np.array(grid, dtype=int)
    bg = _get_bg(g)

    points = [(int(r), int(c)) for r, c in np.argwhere(g != bg)]
    if not points:
        return np.zeros((0, 1), dtype=int)

    components = _connected_components(points)
    components.sort(key=min)

    colors: list[int] = []
    for comp in components:
        for r, c in _trace_component(comp):
            colors.append(int(g[r, c]))
    return np.array(colors, dtype=int).reshape((-1, 1))


def _random_partition(total: int, num_parts: int, min_len: int, max_len: int) -> list[int] | None:
    if num_parts * min_len > total or num_parts * max_len < total:
        return None
    parts = [min_len] * num_parts
    remaining = total - sum(parts)
    for _ in range(remaining):
        eligible = [i for i, part in enumerate(parts) if part < max_len]
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


def _make_path_colors(length: int, palette: list[int], required: int | None, avoid_start: int | None) -> list[int]:
    num_segments = random.randint(2, min(4, length))
    seg_lengths = _random_partition(length, num_segments, min_len=1, max_len=4)
    assert seg_lengths is not None

    first_choices = [c for c in palette if c != avoid_start] or list(palette)
    seg_colors = [random.choice(first_choices)]
    for _ in range(1, num_segments):
        choices = [c for c in palette if c != seg_colors[-1]]
        seg_colors.append(random.choice(choices))
    seg_colors = _ensure_required_color(seg_colors, required)

    colors: list[int] = []
    for seg_len, col in zip(seg_lengths, seg_colors):
        colors.extend([col] * seg_len)
    return colors


def _plan_band_counts(num_paths: int) -> tuple[list[int], list[int]]:
    band_rows = [1, 8, 15]
    active = random.randint((num_paths + 1) // 2, min(num_paths, len(band_rows)))
    rows = sorted(random.sample(band_rows, active))
    counts = [1] * active
    extra = num_paths - active
    slots = list(range(active))
    random.shuffle(slots)
    for idx in slots[:extra]:
        counts[idx] += 1
    return rows, counts


def _place_band_components(
    band_row: int,
    shapes: list[list[tuple[int, int]]],
    H: int,
    W: int,
) -> list[list[tuple[int, int]]]:
    dims = []
    for shape in shapes:
        height = max(r for r, _ in shape) + 1
        width = max(c for _, c in shape) + 1
        assert band_row + height <= H
        dims.append((shape, width))

    placed: list[list[tuple[int, int]]] = []
    if len(dims) == 1:
        shape, width = dims[0]
        col = random.randint(1, W - width - 1)
        placed.append([(band_row + r, col + c) for r, c in shape])
        return placed

    gap = random.randint(2, 4)
    total_width = dims[0][1] + dims[1][1] + gap
    start_col = random.randint(1, W - total_width - 1)
    col = start_col
    for shape, width in dims:
        placed.append([(band_row + r, col + c) for r, c in shape])
        col += width + gap
    return placed


def _gen_attempt() -> np.ndarray | None:
    H, W = 20, 20
    bg = random.randint(0, 9)
    num_paths = random.randint(3, 5)
    palette_size = random.randint(3, num_paths)
    palette = random.sample([c for c in range(10) if c != bg], palette_size)

    total_min = max(20, num_paths * 5)
    total_max = min(30, num_paths * 8)
    target_total = random.randint(total_min, total_max)
    lengths = _random_partition(target_total, num_paths, min_len=5, max_len=8)
    if lengths is None:
        return None

    band_rows, band_counts = _plan_band_counts(num_paths)
    random.shuffle(lengths)

    components: list[list[tuple[int, int]]] = []
    length_idx = 0
    for band_row, band_count in zip(band_rows, band_counts):
        shapes = []
        for _ in range(band_count):
            shapes.append(_sample_relative_path(lengths[length_idx]))
            length_idx += 1
        components.extend(_place_band_components(band_row, shapes, H, W))

    all_points = [pt for comp in components for pt in comp]
    if len(all_points) != len(set(all_points)):
        return None

    traced_components = []
    for comp in components:
        traced_components.append(_trace_component(comp))
    traced_components.sort(key=min)

    grid = np.full((H, W), bg, dtype=int)
    expected: list[int] = []
    needed = list(palette)
    random.shuffle(needed)
    prev_last: int | None = None
    for order in traced_components:
        required = needed.pop() if needed else None
        colors = _make_path_colors(len(order), palette, required=required, avoid_start=prev_last)
        prev_last = colors[-1]
        expected.extend(colors)
        for (r, c), col in zip(order, colors):
            grid[r, c] = col

    comps_from_grid = _connected_components([(int(r), int(c)) for r, c in np.argwhere(grid != bg)])
    if len(comps_from_grid) != num_paths:
        return None
    if any(not (5 <= len(comp) <= 8) for comp in comps_from_grid):
        return None

    expected_out = np.array(expected, dtype=int).reshape((-1, 1))
    actual_out = main(grid)
    if not np.array_equal(actual_out, expected_out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("Failed to generate input after 400 attempts")
