# Combo variant of 7b5033c1 (index 26)
# combo_spec: D2/2+D3/3+D5/2
# Name: 多路径+标记方向+分行输出

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/arc2_solver_select")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import get_bg


# Design analysis for the combo variant
# a) Concretization and trace:
#    - Original task: one long multicolor 4-connected snake. Solver takes all non-bg
#      cells as one component, picks the top-left endpoint, traces once, and reshapes
#      the color sequence into an Nx1 column.
#    - Combo input: 3-5 disconnected short snakes (length 5-8). Near one endpoint of
#      each snake sits one isolated marker pixel of a dedicated color. The marker is
#      not on the snake and is not 4-neighbor to any path cell.
#    - Combo output: one row per snake. For each snake, first decode which endpoint is
#      selected by the nearby marker, trace from that endpoint, then place that traced
#      color sequence into one output row. Finally sort rows by the chosen start
#      endpoint from top to bottom (tie-break left to right) and pad shorter rows with
#      background.
#    - Small trace: suppose there are three snakes. Under the old logic, the solver
#      would wrongly merge markers into the foreground set, would choose `min(endpoints)`
#      inside each component, and would emit a single column. Under the combo rule, the
#      solver must separate path components from singleton markers, choose a start per
#      component from marker geometry, and then pack 3 rows instead of 1 column.
# b) Complexity calibration:
#    - The marker color is fixed and excluded from snake palettes, so the symbol is
#      visually stable across examples.
#    - Snakes stay short and simple, which keeps local path tracing readable, but 3-5
#      components plus per-component direction choice introduces real object grouping.
#    - The generator forces at least one marker to choose the non-default endpoint and
#      forces at least two distinct path lengths, so the D2 and D5 effects stay visible.
# c) Implicit assumptions broken by the fusion:
#    - `mask = g != bg` no longer means "one snake"; it now includes several snakes and
#      several marker singletons, so `main()` must split 4-connected components first.
#    - `ends = [...]` can no longer be computed globally; endpoints must be computed per
#      path component.
#    - `start = min(ends)` is invalid because markers override direction.
#    - `reshape((-1, 1))` is invalid because output is now a padded multi-row table.
#    - A new fusion-specific interaction appears: row order depends on the chosen start
#      endpoint, not merely on the path's top-left occupied cell.
# d) Placement legality:
#    - The generator places snakes in separated row bands, with at most two snakes per
#      band and a horizontal gap, so snakes never overlap or touch 4-neighbor-wise.
#    - Marker candidates are accepted only if they are in bounds, empty, not 4-neighbor
#      to any occupied cell, and strictly closer to the designated endpoint than to any
#      other competing endpoint under the solver's distance rule.
# e) Visual observability:
#    - Humans can see several disconnected snakes and one isolated marker near each.
#      Output row count matches snake count, and row contents visibly follow each local
#      snake rather than one global concatenation. Some rows reverse relative to the
#      top-left endpoint, which exposes the direction marker rule.
# f) Visual cleanliness:
#    - Markers are isolated single pixels rather than overlaid symbols, so they do not
#      occlude the snakes. Band spacing prevents components from blending together.
#    - Padding uses background only; there is no second decoration layer in the output.
# g) Color inferability:
#    - There is no hidden color remapping. Snake colors are preserved exactly, and the
#      only special color is the dedicated marker color. Palettes are kept small and
#      reused across paths, so color itself is not the puzzle; structure is.
# 9) New solver assumptions and generator guarantees:
#    - Path components are exactly the foreground components of size >= 5.
#    - Marker components are singleton cells of color `MARK`.
#    - Every path has exactly one nearest marker under the solver's greedy assignment.
#    - Every path is a simple chain with exactly two endpoints.
#    - At least one path uses the non-default endpoint and has a non-palindromic color
#      sequence, so marker-aware tracing differs from default-endpoint tracing.


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


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    H, W = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        stack = [(r, c)]
        comp: list[tuple[int, int]] = []
        while stack:
            rr, cc = stack.pop()
            comp.append((int(rr), int(cc)))
            for dr, dc in DIR4:
                nr, nc = int(rr + dr), int(cc + dc)
                if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((int(nr), int(nc)))
        comps.append(sorted(comp))
    return comps


def _endpoint_data(component: list[tuple[int, int]]) -> tuple[dict[tuple[int, int], int], list[tuple[int, int]]]:
    pset = set(component)
    deg: dict[tuple[int, int], int] = {}
    ends: list[tuple[int, int]] = []
    for r, c in component:
        d = 0
        for dr, dc in DIR4:
            if (int(r + dr), int(c + dc)) in pset:
                d += 1
        deg[(int(r), int(c))] = int(d)
        if d == 1:
            ends.append((int(r), int(c)))
    return deg, sorted(ends)


def _trace_component(component: list[tuple[int, int]], start: tuple[int, int]) -> list[tuple[int, int]]:
    pset = set(component)
    order: list[tuple[int, int]] = []
    prev: tuple[int, int] | None = None
    cur = (int(start[0]), int(start[1]))
    for _ in range(len(component)):
        order.append(cur)
        r, c = cur
        nbrs = []
        for dr, dc in DIR4:
            q = (int(r + dr), int(c + dc))
            if q in pset and q != prev:
                nbrs.append(q)
        if not nbrs:
            break
        assert len(nbrs) == 1
        prev, cur = cur, nbrs[0]
    assert len(order) == len(component)
    return order


def _assign_markers_to_paths(
    endpoints_by_idx: list[list[tuple[int, int]]],
    markers: list[tuple[int, int]],
) -> dict[int, tuple[int, int] | None]:
    if not markers:
        return {idx: None for idx in range(len(endpoints_by_idx))}

    candidates: list[tuple[tuple[int, int], int, int]] = []
    for path_idx, ends in enumerate(endpoints_by_idx):
        for marker_idx, marker in enumerate(markers):
            dist = min(_dist_key(marker, end) for end in ends)
            candidates.append((dist, int(path_idx), int(marker_idx)))
    candidates.sort()

    assigned: dict[int, tuple[int, int] | None] = {idx: None for idx in range(len(endpoints_by_idx))}
    used_markers: set[int] = set()
    for _dist, path_idx, marker_idx in candidates:
        if assigned[path_idx] is not None or marker_idx in used_markers:
            continue
        assigned[path_idx] = markers[marker_idx]
        used_markers.add(marker_idx)
    return assigned


def _marker_start(ends: list[tuple[int, int]], marker: tuple[int, int] | None) -> tuple[int, int]:
    if marker is None:
        return min(ends)
    scored = [(_dist_key(marker, end), end) for end in ends]
    scored.sort()
    return scored[0][1]


def _pack_rows(rows: list[list[int]], bg: int) -> np.ndarray:
    if not rows:
        return np.zeros((0, 0), dtype=int)
    out_h = len(rows)
    out_w = max(len(row) for row in rows)
    out = np.full((out_h, out_w), int(bg), dtype=int)
    for r, row in enumerate(rows):
        out[int(r), : len(row)] = np.array(row, dtype=int)
    return out


def _extract_path_rows(g: np.ndarray) -> tuple[int, list[tuple[tuple[int, int], list[int]]]]:
    bg = int(get_bg(g))
    mask = g != int(bg)
    if not np.any(mask):
        return bg, []

    comps = _components4(mask)
    path_components = [comp for comp in comps if len(comp) >= 5]
    marker_points = [comp[0] for comp in comps if len(comp) == 1 and int(g[comp[0]]) == int(MARK)]

    endpoints_by_idx: list[list[tuple[int, int]]] = []
    for comp in path_components:
        deg, ends = _endpoint_data(comp)
        assert len(ends) == 2
        assert sum(d == 1 for d in deg.values()) == 2
        assert sum(d == 2 for d in deg.values()) == len(comp) - 2
        endpoints_by_idx.append(ends)

    marker_map = _assign_markers_to_paths(endpoints_by_idx, marker_points)
    rows: list[tuple[tuple[int, int], list[int]]] = []
    for idx, comp in enumerate(path_components):
        start = _marker_start(endpoints_by_idx[idx], marker_map[idx])
        order = _trace_component(comp, start)
        rows.append((start, [int(g[r, c]) for r, c in order]))
    rows.sort(key=lambda item: item[0])
    return bg, rows


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    - Non-bg cells form several simple snake components plus isolated marker pixels.
    - Each marker selects which endpoint of its nearby snake is the traversal start.
    - Each traced snake becomes one output row.
    - Rows are sorted by the chosen start endpoint and padded with background.
    """
    g = np.array(grid, dtype=int)
    bg, rows = _extract_path_rows(g)
    return _pack_rows([row for _start, row in rows], bg)


def _random_partition(total: int, num_parts: int, min_len: int, max_len: int) -> list[int] | None:
    if num_parts * min_len > total or num_parts * max_len < total:
        return None
    for _ in range(80):
        parts = [int(min_len)] * int(num_parts)
        remaining = int(total - sum(parts))
        for _ in range(remaining):
            eligible = [i for i, part in enumerate(parts) if part < max_len]
            if not eligible:
                break
            parts[random.choice(eligible)] += 1
        if sum(parts) != total:
            continue
        if len(set(parts)) < 2:
            continue
        random.shuffle(parts)
        return [int(part) for part in parts]
    return None


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


def _plan_band_counts(num_paths: int) -> tuple[list[int], list[int]]:
    band_rows = [2, 10, 18]
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
        assert band_row + height <= H - 1
        dims.append((shape, width))

    placed: list[list[tuple[int, int]]] = []
    if len(dims) == 1:
        shape, width = dims[0]
        col = random.randint(2, W - width - 3)
        placed.append([(int(band_row + r), int(col + c)) for r, c in shape])
        return placed

    gap = random.randint(5, 7)
    total_width = dims[0][1] + dims[1][1] + gap
    start_col = random.randint(2, W - total_width - 3)
    col = start_col
    for shape, width in dims:
        placed.append([(int(band_row + r), int(col + c)) for r, c in shape])
        col += width + gap
    return placed


def _make_path_colors(length: int, palette: list[int]) -> list[int] | None:
    for _ in range(80):
        num_segments = random.randint(2, min(4, length))
        seg_lengths = _random_partition(length, num_segments, min_len=1, max_len=4)
        if seg_lengths is None:
            continue

        seg_colors = [random.choice(palette)]
        for _ in range(1, num_segments):
            choices = [c for c in palette if c != seg_colors[-1]]
            seg_colors.append(random.choice(choices))

        colors: list[int] = []
        for seg_len, col in zip(seg_lengths, seg_colors):
            colors.extend([int(col)] * int(seg_len))
        if len(set(colors)) < 2:
            continue
        if colors == colors[::-1]:
            continue
        return colors
    return None


def _choose_marker_cell(
    chosen_end: tuple[int, int],
    other_end: tuple[int, int],
    other_endpoints: list[tuple[int, int]],
    occupied: set[tuple[int, int]],
    H: int,
    W: int,
) -> tuple[int, int] | None:
    offsets = list(MARK_OFFSETS)
    random.shuffle(offsets)
    offsets.sort(key=lambda d: (max(abs(d[0]), abs(d[1])), abs(d[0]) + abs(d[1])))
    for dr, dc in offsets:
        nr, nc = int(chosen_end[0] + dr), int(chosen_end[1] + dc)
        if not (0 <= nr < H and 0 <= nc < W):
            continue
        if (nr, nc) in occupied:
            continue
        bad_neighbor = False
        for ddr, ddc in DIR4:
            if (int(nr + ddr), int(nc + ddc)) in occupied:
                bad_neighbor = True
                break
        if bad_neighbor:
            continue
        if _dist_key((nr, nc), chosen_end) >= _dist_key((nr, nc), other_end):
            continue
        if any(_dist_key((nr, nc), chosen_end) >= _dist_key((nr, nc), end) for end in other_endpoints):
            continue
        return nr, nc
    return None


def _default_rows_without_markers(g: np.ndarray, bg: int) -> np.ndarray:
    comps = [comp for comp in _components4(g != int(bg)) if len(comp) >= 5]
    rows: list[tuple[tuple[int, int], list[int]]] = []
    for comp in comps:
        _deg, ends = _endpoint_data(comp)
        start = min(ends)
        order = _trace_component(comp, start)
        rows.append((start, [int(g[r, c]) for r, c in order]))
    rows.sort(key=lambda item: item[0])
    return _pack_rows([row for _start, row in rows], bg)


def _gen_attempt() -> np.ndarray | None:
    H, W = 24, 24
    bg = random.choice([c for c in range(10) if c != MARK])
    num_paths = random.randint(3, 5)
    palette_size = random.randint(3, 5)
    palette = random.sample([c for c in range(10) if c not in (bg, MARK)], palette_size)

    lengths = [random.randint(5, 8) for _ in range(num_paths)]
    if len(set(lengths)) < 2:
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

    specs = []
    reversed_count = 0
    for comp in components:
        _deg, ends = _endpoint_data(comp)
        if len(ends) != 2:
            return None
        start = random.choice(ends)
        if start != min(ends):
            reversed_count += 1
        specs.append({"component": comp, "ends": ends, "start": start})
    if reversed_count == 0:
        idx = random.randrange(len(specs))
        ends = specs[idx]["ends"]
        specs[idx]["start"] = ends[1]
        reversed_count = 1

    occupied = set(all_points)
    all_endpoints = [end for spec in specs for end in spec["ends"]]
    markers: list[tuple[int, int]] = []
    for spec in specs:
        chosen_end = spec["start"]
        other_end = spec["ends"][0] if chosen_end == spec["ends"][1] else spec["ends"][1]
        other_endpoints = [end for end in all_endpoints if end not in spec["ends"]]
        marker = _choose_marker_cell(chosen_end, other_end, other_endpoints, occupied, H, W)
        if marker is None:
            return None
        markers.append(marker)
        occupied.add(marker)
        spec["marker"] = marker

    grid = np.full((H, W), int(bg), dtype=int)
    for spec in specs:
        colors = _make_path_colors(len(spec["component"]), palette)
        if colors is None:
            return None
        if spec["start"] != min(spec["ends"]) and colors == colors[::-1]:
            return None
        spec["colors"] = colors
        order = _trace_component(spec["component"], spec["start"])
        spec["order"] = order
        for (r, c), col in zip(order, colors):
            grid[int(r), int(c)] = int(col)

    for marker in markers:
        grid[marker] = int(MARK)

    expected_rows = sorted(
        ((spec["start"], [int(grid[r, c]) for r, c in spec["order"]]) for spec in specs),
        key=lambda item: item[0],
    )
    expected_out = _pack_rows([row for _start, row in expected_rows], bg)
    actual_out = main(grid)
    if not np.array_equal(actual_out, expected_out):
        return None

    if np.array_equal(actual_out, _default_rows_without_markers(grid, bg)):
        return None

    recovered_bg, recovered_rows = _extract_path_rows(grid)
    if recovered_bg != bg:
        return None
    if len(recovered_rows) != num_paths:
        return None
    if len({len(row) for _start, row in recovered_rows}) < 2:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(600):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("Failed to generate input after 600 attempts")
