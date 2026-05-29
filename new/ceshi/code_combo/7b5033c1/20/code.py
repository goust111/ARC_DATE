# Combo variant of 7b5033c1 (index 20)
# combo_spec: D2/2+D3/3
# Name: 多路径+标记起点
#
# Design analysis
# a) Concrete behavior and trace:
#    Original 7b5033c1 sees one long snake, picks the lexicographically smaller
#    endpoint, and emits one vertical strip. This combo changes both the object
#    count and the way each object is read.
#    Small fused example:
#    - path A cells form a 5-pixel chain near the top, with endpoints a0 and a1.
#      A detached red marker sits diagonally beside a1, so A must be read from a1.
#    - path B is lower in the grid with endpoints b0 and b1; its red marker sits
#      beside b0, so B is read from b0.
#    - Original logic would wrongly treat all non-bg pixels as one set and would
#      also ignore the marker direction cue. Fused logic must instead:
#      1. split foreground into components,
#      2. identify the 3-5 chain components and the detached marker singletons,
#      3. sort chain components top-to-bottom / left-to-right,
#      4. for each chain, use its nearby marker to choose which endpoint is first,
#      5. trace that chain and concatenate its colors with the next chain.
#    The visible difference from D3-only is that some local chains are reversed
#    relative to their top-left endpoint; the visible difference from D2-only is
#    that there are several independently marked chains rather than one.
#
# b) Complexity calibration:
#    Markers use one fixed special color and are detached singleton pixels, so the
#    clue is visually stable. Paths are short (5-8 cells) and there are 3-5 of
#    them, which keeps the object-sorting problem real without making the board
#    cluttered. The generator forces at least one chain to start from its non-
#    default endpoint, so the marker dimension is active rather than decorative.
#
# c) Hidden assumptions broken from the original main():
#    - `g != bg` no longer means one connected snake; it now contains many chain
#      components plus marker singleton components.
#    - Degree counting on all non-bg cells together is invalid because endpoints
#      must be computed per chain.
#    - `start = min(ends)` is invalid because each chain's direction is controlled
#      by its own marker.
#    - A single traversal loop over all foreground cells is invalid because chains
#      must be ordered globally but traced independently.
#    The revised solver therefore isolates components first, pairs each path with
#    one marker, and only then traces each chain.
#
# d) Placement legality:
#    Generator places paths in separated row bands with generous column gaps, so
#    paths never overlap or touch 4-neighbor-wise. Every marker is placed on a
#    background cell, stays in-bounds, is not 4-neighbor to any path or marker,
#    and is strictly closer to its intended endpoint than to every competing
#    endpoint under the solver's distance rule.
#
# e) Visual observability:
#    Across examples, humans can see several small snakes, one red dot per snake,
#    and one long output column. Because at least one snake is marker-reversed in
#    each sample, the examples show that the red dot is not optional decoration:
#    it decides which end's color appears first for that local snake.
#
# f) Visual cleanliness:
#    The board uses separated row bands, at most two snakes per band, and markers
#    sit just outside snake endpoints. Nothing overlaps and no region carries more
#    than one type of signal, so components remain easy to individuate by eye.
#
# g) Color inference:
#    There is no hidden recoloring rule. Snake pixels keep their original colors,
#    markers all share the same special color, and output is only a reordering of
#    observed snake colors. The rule is inferable from geometry plus the marker.
#
# h) Quantitative anti-degeneration target:
#    Valid generated samples must satisfy `flipped_paths >= 1`, where a flipped
#    path is one whose marked start differs from `min(endpoints)`. We also reject
#    samples whose expected output equals the no-marker baseline over the paths.
#
# i) New solver assumptions and generator guarantees:
#    - Solver assumes chain components have size > 1 and marker components are
#      singleton cells of the special marker color. Generator guarantees exactly
#      3-5 path components and one singleton marker per path.
#    - Solver assumes each marker can be paired to one path by nearest-endpoint
#      distance. Generator enforces strict nearest-endpoint uniqueness.
#    - Solver assumes each path is a simple 4-connected chain with exactly two
#      endpoints. Generator uses fixed short snake templates and spacing to ensure
#      this invariant.

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

try:
    from arc_utils import get_bg
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[4] / "arc2_solver_select"))
    from arc_utils import get_bg


MARK = 2
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
MARK_OFFSETS = [
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
    (-2, -1),
    (-2, 1),
    (2, -1),
    (2, 1),
    (-1, -2),
    (1, -2),
    (-1, 2),
    (1, 2),
    (-2, -2),
    (-2, 2),
    (2, -2),
    (2, 2),
    (-2, 0),
    (2, 0),
    (0, -2),
    (0, 2),
]
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


def _dist_key(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return (
        max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1]))),
        abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1])),
    )


def _components4(points: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    pset = set(points)
    seen: set[tuple[int, int]] = set()
    comps: list[list[tuple[int, int]]] = []
    for start in sorted(points):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        comp: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            comp.append((int(r), int(c)))
            for dr, dc in DIR4:
                nxt = (int(r + dr), int(c + dc))
                if nxt in pset and nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        comps.append(sorted(comp))
    return comps


def _component_endpoints(component: list[tuple[int, int]]) -> list[tuple[int, int]]:
    pset = set(component)
    ends: list[tuple[int, int]] = []
    for r, c in component:
        deg = sum((int(r + dr), int(c + dc)) in pset for dr, dc in DIR4)
        if deg == 1:
            ends.append((int(r), int(c)))
    return sorted(ends)


def _trace_component(component: list[tuple[int, int]], start: tuple[int, int]) -> list[tuple[int, int]]:
    pts = sorted(component)
    if len(pts) <= 1:
        return pts

    pset = set(pts)
    order: list[tuple[int, int]] = []
    prev: tuple[int, int] | None = None
    cur = (int(start[0]), int(start[1]))
    visited: set[tuple[int, int]] = set()
    while cur not in visited and cur in pset:
        visited.add(cur)
        order.append(cur)
        r, c = cur
        nbrs = sorted(
            (int(r + dr), int(c + dc))
            for dr, dc in DIR4
            if (int(r + dr), int(c + dc)) in pset and (int(r + dr), int(c + dc)) != prev
        )
        if not nbrs:
            break
        prev, cur = cur, nbrs[0]

    if len(order) == len(pts):
        return order

    # Fallback for malformed inputs: keep deterministic order instead of crashing.
    leftovers = [pt for pt in pts if pt not in visited]
    return order + leftovers


def _path_marker_score(component: list[tuple[int, int]], marker: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    ends = _component_endpoints(component)
    if not ends:
        anchor = min(component)
        return _dist_key(marker, anchor), anchor
    best_end = min(ends, key=lambda end: (_dist_key(marker, end), end))
    return _dist_key(marker, best_end), best_end


def _assign_markers(
    path_components: list[list[tuple[int, int]]],
    marker_pts: list[tuple[int, int]],
) -> dict[int, tuple[int, int] | None]:
    assignments = {idx: None for idx in range(len(path_components))}
    if not marker_pts:
        return assignments

    scored_pairs: list[tuple[tuple[int, int], int, int]] = []
    for path_idx, comp in enumerate(path_components):
        for marker_idx, marker in enumerate(marker_pts):
            score, _best_end = _path_marker_score(comp, marker)
            scored_pairs.append((score, path_idx, marker_idx))

    used_paths: set[int] = set()
    used_markers: set[int] = set()
    for _score, path_idx, marker_idx in sorted(scored_pairs):
        if path_idx in used_paths or marker_idx in used_markers:
            continue
        used_paths.add(path_idx)
        used_markers.add(marker_idx)
        assignments[path_idx] = marker_pts[marker_idx]
    return assignments


def _default_start(component: list[tuple[int, int]]) -> tuple[int, int]:
    ends = _component_endpoints(component)
    return min(ends) if ends else min(component)


def _marked_start(component: list[tuple[int, int]], marker: tuple[int, int] | None) -> tuple[int, int]:
    ends = _component_endpoints(component)
    if not ends:
        return min(component)
    if marker is None:
        return min(ends)
    return min(ends, key=lambda end: (_dist_key(marker, end), end))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    - Non-bg cells form 3-5 disconnected short snake paths and one detached marker
      singleton near one endpoint of each path.
    - Sort path components by their topmost/leftmost occupied cell.
    - For each path, choose the endpoint nearest to its paired marker as the start.
    - Trace the path from that start and concatenate all path colors into one column.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    points = [(int(r), int(c)) for r, c in np.argwhere(g != bg)]
    if not points:
        return np.zeros((0, 1), dtype=int)

    comps = _components4(points)
    path_components = [comp for comp in comps if len(comp) > 1]
    marker_pts = [comp[0] for comp in comps if len(comp) == 1 and int(g[comp[0]]) == MARK]
    path_components.sort(key=min)

    marker_map = _assign_markers(path_components, marker_pts)
    colors: list[int] = []
    for idx, comp in enumerate(path_components):
        start = _marked_start(comp, marker_map[idx])
        for r, c in _trace_component(comp, start):
            colors.append(int(g[r, c]))
    return np.array(colors, dtype=int).reshape((-1, 1))


def _random_partition(total: int, num_parts: int, min_len: int, max_len: int) -> list[int] | None:
    if num_parts * min_len > total or num_parts * max_len < total:
        return None
    parts = [min_len] * num_parts
    remaining = total - sum(parts)
    for _ in range(remaining):
        eligible = [idx for idx, value in enumerate(parts) if value < max_len]
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
        colors.extend([int(col)] * int(seg_len))
    return colors


def _plan_band_counts(num_paths: int) -> tuple[list[int], list[int]]:
    band_rows = [2, 9, 16]
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
    h: int,
    w: int,
) -> list[list[tuple[int, int]]]:
    dims: list[tuple[list[tuple[int, int]], int]] = []
    for shape in shapes:
        height = max(r for r, _ in shape) + 1
        width = max(c for _, c in shape) + 1
        if band_row + height > h - 2:
            raise ValueError("Band placement exceeds board height")
        dims.append((shape, width))

    placed: list[list[tuple[int, int]]] = []
    if len(dims) == 1:
        shape, width = dims[0]
        col = random.randint(2, w - width - 3)
        placed.append([(band_row + r, col + c) for r, c in shape])
        return placed

    gap = random.randint(4, 7)
    total_width = sum(width for _shape, width in dims) + gap * (len(dims) - 1)
    start_col = random.randint(2, w - total_width - 3)
    col = start_col
    for shape, width in dims:
        placed.append([(band_row + r, col + c) for r, c in shape])
        col += width + gap
    return placed


def _choose_marker_cell(
    chosen_end: tuple[int, int],
    own_other_end: tuple[int, int],
    other_endpoints: list[tuple[int, int]],
    blocked: set[tuple[int, int]],
    h: int,
    w: int,
) -> tuple[int, int] | None:
    offsets = list(MARK_OFFSETS)
    random.shuffle(offsets)
    offsets.sort(key=lambda d: (max(abs(d[0]), abs(d[1])), abs(d[0]) + abs(d[1])))
    for dr, dc in offsets:
        nr, nc = int(chosen_end[0] + dr), int(chosen_end[1] + dc)
        cand = (nr, nc)
        if not (0 <= nr < h and 0 <= nc < w):
            continue
        if cand in blocked:
            continue
        if any((int(nr + ddr), int(nc + ddc)) in blocked for ddr, ddc in DIR4):
            continue

        own_score = _dist_key(cand, chosen_end)
        if own_score >= _dist_key(cand, own_other_end):
            continue

        if any(own_score >= _dist_key(cand, other_end) for other_end in other_endpoints):
            continue
        return cand
    return None


def _default_output(grid: np.ndarray, path_components: list[list[tuple[int, int]]]) -> np.ndarray:
    colors: list[int] = []
    for comp in sorted(path_components, key=min):
        for r, c in _trace_component(comp, _default_start(comp)):
            colors.append(int(grid[r, c]))
    return np.array(colors, dtype=int).reshape((-1, 1))


def _gen_attempt() -> np.ndarray | None:
    h, w = 22, 22
    bg = random.choice([c for c in range(10) if c != MARK])
    num_paths = random.randint(3, 5)
    palette_size = random.randint(3, min(5, num_paths + 1))
    palette = random.sample([c for c in range(10) if c not in (bg, MARK)], palette_size)

    total_min = max(20, num_paths * 5)
    total_max = min(30, num_paths * 8)
    total_len = random.randint(total_min, total_max)
    lengths = _random_partition(total_len, num_paths, min_len=5, max_len=8)
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
        try:
            components.extend(_place_band_components(band_row, shapes, h, w))
        except ValueError:
            return None

    all_path_cells = [pt for comp in components for pt in comp]
    if len(all_path_cells) != len(set(all_path_cells)):
        return None

    ordered_components = sorted([sorted(comp) for comp in components], key=min)
    endpoints_per_path = [_component_endpoints(comp) for comp in ordered_components]
    if any(len(ends) != 2 for ends in endpoints_per_path):
        return None

    flip_indices = set(random.sample(range(num_paths), random.randint(1, num_paths)))
    marker_pts: list[tuple[int, int]] = []
    marked_starts: list[tuple[int, int]] = []
    blocked = set(all_path_cells)
    for idx, comp in enumerate(ordered_components):
        ends = endpoints_per_path[idx]
        default_start = min(ends)
        non_default = ends[1] if ends[0] == default_start else ends[0]
        marked_start = non_default if idx in flip_indices else random.choice(ends)
        marked_starts.append(marked_start)
        other_end = ends[0] if marked_start == ends[1] else ends[1]
        other_endpoints = [end for j, pair in enumerate(endpoints_per_path) if j != idx for end in pair]
        marker = _choose_marker_cell(marked_start, other_end, other_endpoints, blocked, h, w)
        if marker is None:
            return None
        marker_pts.append(marker)
        blocked.add(marker)

    flipped_paths = sum(start != min(ends) for start, ends in zip(marked_starts, endpoints_per_path))
    if flipped_paths < 1:
        return None

    grid = np.full((h, w), bg, dtype=int)
    expected: list[int] = []
    needed = list(palette)
    random.shuffle(needed)
    prev_last: int | None = None
    for comp, start in zip(ordered_components, marked_starts):
        order = _trace_component(comp, start)
        required = needed.pop() if needed else None
        colors = _make_path_colors(len(order), palette, required=required, avoid_start=prev_last)
        prev_last = colors[-1]
        expected.extend(colors)
        for (r, c), col in zip(order, colors):
            grid[r, c] = int(col)

    for marker in marker_pts:
        grid[marker] = MARK

    comps_from_grid = _components4([(int(r), int(c)) for r, c in np.argwhere(grid != bg)])
    path_count = sum(len(comp) > 1 for comp in comps_from_grid)
    marker_count = sum(len(comp) == 1 and int(grid[comp[0]]) == MARK for comp in comps_from_grid)
    if path_count != num_paths or marker_count != num_paths:
        return None

    expected_out = np.array(expected, dtype=int).reshape((-1, 1))
    default_out = _default_output(grid, ordered_components)
    if np.array_equal(expected_out, default_out):
        return None

    actual_out = main(grid)
    if not np.array_equal(actual_out, expected_out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(600):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("Failed to generate input after 600 attempts")
