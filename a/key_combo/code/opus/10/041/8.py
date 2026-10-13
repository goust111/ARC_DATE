from __future__ import annotations

from collections import defaultdict, deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
PANEL = 8

LABEL_COLORS = (BLUE, RED, GREEN)
SHAPE_LIBRARY = (
    frozenset({(0, 0), (0, 1)}),
    frozenset({(0, 0), (1, 0)}),
    frozenset({(0, 0), (0, 1), (1, 0)}),
    frozenset({(0, 0), (0, 1), (1, 1)}),
    frozenset({(0, 0), (1, 0), (1, 1)}),
    frozenset({(0, 1), (1, 0), (1, 1)}),
    frozenset({(0, 0), (0, 1), (1, 0), (1, 1)}),
)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(coords: list[tuple[int, int]] | set[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    r0, c0, _r1, _c1 = _bbox(coords)
    return frozenset((r - r0, c - c0) for r, c in coords)


def _components_from_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _component_profile(coords: list[tuple[int, int]]) -> dict[str, object]:
    rows: dict[int, tuple[int, int]] = {}
    cols: dict[int, tuple[int, int]] = {}
    for r, c in coords:
        if r not in rows:
            rows[r] = (c, c)
        else:
            rows[r] = (min(rows[r][0], c), max(rows[r][1], c))
        if c not in cols:
            cols[c] = (r, r)
        else:
            cols[c] = (min(cols[c][0], r), max(cols[c][1], r))
    return {
        "coords": tuple(sorted(coords)),
        "coord_set": set(coords),
        "bbox": _bbox(coords),
        "rows": rows,
        "cols": cols,
        "shape": _normalize(coords),
    }


def _panel_bbox_and_component(grid: np.ndarray) -> tuple[tuple[int, int, int, int], set[tuple[int, int]]]:
    comps = _components_from_mask(grid != BG)
    candidates = [comp for comp in comps if any(int(grid[r, c]) == PANEL for r, c in comp)]
    assert len(candidates) == 1, "expected exactly one teal legend panel"
    comp = candidates[0]
    return _bbox(comp), set(comp)


def _extract_legend(
    grid: np.ndarray,
) -> tuple[dict[frozenset[tuple[int, int]], int], list[int], tuple[int, int, int, int]]:
    panel_box, _panel_component = _panel_bbox_and_component(grid)
    r0, c0, r1, c1 = panel_box
    panel = grid[r0 : r1 + 1, c0 : c1 + 1]

    hole_entries = []
    for comp in _components_from_mask(panel == BG):
        if any(rr == 0 or cc == 0 or rr == panel.shape[0] - 1 or cc == panel.shape[1] - 1 for rr, cc in comp):
            continue
        neighbor_colors: set[int] = set()
        neighbor_positions: list[tuple[int, int]] = []
        for rr, cc in comp:
            for nr, nc in _neighbors4(rr, cc):
                if 0 <= nr < panel.shape[0] and 0 <= nc < panel.shape[1]:
                    val = int(panel[nr, nc])
                    if val not in {BG, PANEL}:
                        neighbor_colors.add(val)
                        neighbor_positions.append((nr, nc))
        assert len(neighbor_colors) == 1, "each hole must have exactly one adjacent label color"
        color = next(iter(neighbor_colors))
        assert color in LABEL_COLORS
        label_key = min(neighbor_positions)
        hole_entries.append((label_key, _normalize(comp), color))

    hole_entries.sort(key=lambda item: item[0])
    assert 2 <= len(hole_entries) <= 3, "legend should contain 2-3 entries"

    shape_to_color: dict[frozenset[tuple[int, int]], int] = {}
    priority: list[int] = []
    for _label_key, shape, color in hole_entries:
        assert shape not in shape_to_color, "legend hole shapes must be unique"
        assert color not in priority, "legend colors must be unique"
        shape_to_color[shape] = color
        priority.append(color)
    return shape_to_color, priority, panel_box


def _collect_scattered_components(
    grid: np.ndarray,
    panel_box: tuple[int, int, int, int],
    shape_to_color: dict[frozenset[tuple[int, int]], int],
) -> dict[int, list[dict[str, object]]]:
    r0, c0, r1, c1 = panel_box
    mask = grid == YELLOW
    mask[r0 : r1 + 1, c0 : c1 + 1] = False
    color_groups: dict[int, list[dict[str, object]]] = defaultdict(list)
    for comp in _components_from_mask(mask):
        shape = _normalize(comp)
        assert shape in shape_to_color, "every scattered yellow shape must match a legend hole"
        logical_color = shape_to_color[shape]
        color_groups[logical_color].append(_component_profile(comp))
    total = sum(len(v) for v in color_groups.values())
    assert 4 <= total <= 8, "expected 4-8 scattered shapes"
    return color_groups


def _pair_fill_cells(
    left_or_top: dict[str, object],
    right_or_bottom: dict[str, object],
    base: np.ndarray,
) -> set[tuple[int, int]]:
    fill: set[tuple[int, int]] = set()

    a_box = left_or_top["bbox"]
    b_box = right_or_bottom["bbox"]
    if a_box[1] <= b_box[1]:
        left, right = left_or_top, right_or_bottom
    else:
        left, right = right_or_bottom, left_or_top
    for row in sorted(set(left["rows"]).intersection(right["rows"])):
        left_end = left["rows"][row][1]
        right_start = right["rows"][row][0]
        if right_start - left_end <= 1:
            continue
        if np.all(base[row, left_end + 1 : right_start] == BG):
            for col in range(left_end + 1, right_start):
                fill.add((row, col))

    if a_box[0] <= b_box[0]:
        top, bottom = left_or_top, right_or_bottom
    else:
        top, bottom = right_or_bottom, left_or_top
    for col in sorted(set(top["cols"]).intersection(bottom["cols"])):
        top_end = top["cols"][col][1]
        bottom_start = bottom["cols"][col][0]
        if bottom_start - top_end <= 1:
            continue
        if np.all(base[top_end + 1 : bottom_start, col] == BG):
            for row in range(top_end + 1, bottom_start):
                fill.add((row, col))

    return fill


def _all_fill_paths(
    base: np.ndarray, color_groups: dict[int, list[dict[str, object]]]
) -> dict[int, set[tuple[int, int]]]:
    fill_by_color: dict[int, set[tuple[int, int]]] = defaultdict(set)
    for color, comps in color_groups.items():
        for i in range(len(comps)):
            for j in range(i + 1, len(comps)):
                fill_by_color[color] |= _pair_fill_cells(comps[i], comps[j], base)
    return fill_by_color


def _apply_priority(
    base: np.ndarray, fill_by_color: dict[int, set[tuple[int, int]]], priority: list[int]
) -> np.ndarray:
    out = base.copy()
    for color in reversed(priority):
        for r, c in sorted(fill_by_color.get(color, set())):
            out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Look up each yellow shape in the teal legend, recolor it to its logical color, then fill unobstructed gaps between opposing same-color shapes.

    Input structure:
    - The background is black.
    - A teal legend panel near the upper-left contains 2-3 carved black hole-shapes, each with a neighboring single-pixel color label that defines a shape-to-color mapping.
    - Outside the panel, 4-8 scattered shapes are all drawn in yellow, but each one matches one legend hole by translation-normalized shape.

    Rule:
    - Extract every hole-shape from the panel and map its normalized shape to the adjacent label color.
    - Recolor each scattered yellow shape to the logical color looked up from the legend.
    - For every pair of scattered shapes with the same logical color, fill every unobstructed horizontal or vertical background span between their opposing occupied rows or columns.
    - If fills of different logical colors cross, the color whose legend entry appears higher in the panel overrides lower entries at the crossing cells.
    - Keep the legend panel itself unchanged.

    Key insight:
    - The yellow scattered shapes are visually indistinguishable by color, so the legend is not decorative: shape lookup reveals the hidden grouping, and only then can the correct gap-filling pairs be determined.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    shape_to_color, priority, panel_box = _extract_legend(g)
    color_groups = _collect_scattered_components(g, panel_box, shape_to_color)

    base = g.copy()
    for logical_color, comps in color_groups.items():
        for comp in comps:
            rows, cols = zip(*comp["coords"])
            base[list(rows), list(cols)] = logical_color

    fill_by_color = _all_fill_paths(base, color_groups)
    out = _apply_priority(base, fill_by_color, priority)

    r0, c0, r1, c1 = panel_box
    out[r0 : r1 + 1, c0 : c1 + 1] = g[r0 : r1 + 1, c0 : c1 + 1]
    return out


def _shape_cells(shape: frozenset[tuple[int, int]], anchor: tuple[int, int]) -> list[tuple[int, int]]:
    ar, ac = anchor
    return [(ar + r, ac + c) for r, c in shape]


def _shape_halo(cells: list[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    halo: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < h and 0 <= cc < w:
                    halo.add((rr, cc))
    return halo


def _shape_size(shape: frozenset[tuple[int, int]]) -> tuple[int, int]:
    r0, c0, r1, c1 = _bbox(shape)
    return r1 - r0 + 1, c1 - c0 + 1


def _build_panel(
    grid: np.ndarray,
    legend_shapes: list[frozenset[tuple[int, int]]],
    legend_colors: list[int],
    rng: np.random.Generator,
) -> tuple[int, int, int, int]:
    panel_h = 3 * len(legend_shapes) + 1
    panel_w = 5
    grid[:panel_h, :panel_w] = PANEL
    for idx, (shape, color) in enumerate(zip(legend_shapes, legend_colors)):
        slot_r = 1 + 3 * idx
        hole_cells = [(slot_r + r, 1 + c) for r, c in shape]
        for r, c in hole_cells:
            grid[r, c] = BG
        right_rows = sorted({slot_r + r for r, c in shape if c == 1})
        left_rows = sorted({slot_r + r for r, c in shape if c == 0})
        if right_rows and (not left_rows or rng.random() < 0.5):
            label_r = int(rng.choice(right_rows))
            label_c = 3
        else:
            label_r = int(rng.choice(left_rows))
            label_c = 0
        grid[label_r, label_c] = color
    return 0, 0, panel_h - 1, panel_w - 1


def _cells_to_profile(cells: list[tuple[int, int]]) -> dict[str, object]:
    return _component_profile(cells)


def _can_place_cells(
    cells: list[tuple[int, int]],
    occupied_halo: set[tuple[int, int]],
    h: int,
    w: int,
    work_r0: int,
    panel_box: tuple[int, int, int, int],
) -> bool:
    pr0, pc0, pr1, pc1 = panel_box
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if r < work_r0:
            return False
        if pr0 <= r <= pr1 and pc0 <= c <= pc1:
            return False
        if (r, c) in occupied_halo:
            return False
    return True


def _pair_path_from_cells(
    left_cells: list[tuple[int, int]], right_cells: list[tuple[int, int]]
) -> set[tuple[int, int]]:
    dummy_h = max(max(r for r, _ in left_cells), max(r for r, _ in right_cells)) + 1
    dummy_w = max(max(c for _, c in left_cells), max(c for _, c in right_cells)) + 1
    base = np.zeros((dummy_h, dummy_w), dtype=int)
    for r, c in left_cells + right_cells:
        base[r, c] = 9
    return _pair_fill_cells(_cells_to_profile(left_cells), _cells_to_profile(right_cells), base)


def _place_pair(
    shape: frozenset[tuple[int, int]],
    orientation: str,
    occupied_halo: set[tuple[int, int]],
    occupied_core: set[tuple[int, int]],
    reserved_paths: list[set[tuple[int, int]]],
    h: int,
    w: int,
    work_r0: int,
    work_c0: int,
    panel_box: tuple[int, int, int, int],
    rng: np.random.Generator,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], set[tuple[int, int]]] | None:
    shape_rows = sorted({r for r, _ in shape})
    shape_cols = sorted({c for _, c in shape})
    for _ in range(400):
        if orientation == "H":
            local_row = int(rng.choice(shape_rows))
            anchor_r = int(rng.integers(work_r0, h - 1))
            if anchor_r + 1 >= h:
                continue
            left_c = int(rng.integers(work_c0, max(work_c0 + 1, w - 8)))
            right_c = int(rng.integers(left_c + 4, w - 1))
            if right_c + 1 >= w:
                continue
            left_cells = _shape_cells(shape, (anchor_r, left_c))
            right_cells = _shape_cells(shape, (anchor_r, right_c))
            target_row = anchor_r + local_row
            if target_row >= h:
                continue
        else:
            local_col = int(rng.choice(shape_cols))
            anchor_c = int(rng.integers(work_c0, w - 1))
            if anchor_c + 1 >= w:
                continue
            top_r = int(rng.integers(work_r0, max(work_r0 + 1, h - 8)))
            bottom_r = int(rng.integers(top_r + 4, h - 1))
            if bottom_r + 1 >= h:
                continue
            left_cells = _shape_cells(shape, (top_r, anchor_c))
            right_cells = _shape_cells(shape, (bottom_r, anchor_c))
            target_col = anchor_c + local_col
            if target_col >= w:
                continue

        if not _can_place_cells(left_cells, occupied_halo, h, w, work_r0, panel_box):
            continue
        if not _can_place_cells(right_cells, occupied_halo | _shape_halo(left_cells, h, w), h, w, work_r0, panel_box):
            continue

        path = _pair_path_from_cells(left_cells, right_cells)
        if not path:
            continue
        if path & occupied_core:
            continue
        if any(path & other for other in reserved_paths):
            continue
        return left_cells, right_cells, path
    return None


def _place_crossing_pairs(
    specs: list[tuple[int, frozenset[tuple[int, int]], str]],
    occupied_halo: set[tuple[int, int]],
    occupied_core: set[tuple[int, int]],
    h: int,
    w: int,
    work_r0: int,
    work_c0: int,
    panel_box: tuple[int, int, int, int],
    rng: np.random.Generator,
) -> tuple[
    dict[int, list[list[tuple[int, int]]]],
    dict[int, set[tuple[int, int]]],
    set[tuple[int, int]],
    set[tuple[int, int]],
] | None:
    assert {spec[2] for spec in specs} == {"H", "V"}
    h_spec = next(spec for spec in specs if spec[2] == "H")
    v_spec = next(spec for spec in specs if spec[2] == "V")
    h_color, h_shape, _ = h_spec
    v_color, v_shape, _ = v_spec
    h_h, h_w = _shape_size(h_shape)
    v_h, v_w = _shape_size(v_shape)
    h_rows = sorted({r for r, _ in h_shape})
    v_cols = sorted({c for _, c in v_shape})

    left_h_c = work_c0
    right_h_c = w - h_w
    top_v_r = work_r0
    bottom_v_r = h - v_h
    if right_h_c - left_h_c < 4 or bottom_v_r - top_v_r < 4:
        return None

    for _ in range(200):
        h_local_row = int(rng.choice(h_rows))
        v_local_col = int(rng.choice(v_cols))

        h_path_lo = left_h_c + max(c for r, c in h_shape if r == h_local_row) + 1
        h_path_hi = right_h_c + min(c for r, c in h_shape if r == h_local_row) - 1
        v_path_lo = top_v_r + max(r for r, c in v_shape if c == v_local_col) + 1
        v_path_hi = bottom_v_r + min(r for r, c in v_shape if c == v_local_col) - 1
        cross_c_lo = max(work_c0 + v_local_col, h_path_lo)
        cross_c_hi = min(w - v_w + v_local_col, h_path_hi)
        cross_r_lo = max(work_r0 + h_local_row, v_path_lo)
        cross_r_hi = min(h - h_h + h_local_row, v_path_hi)
        if cross_c_lo > cross_c_hi or cross_r_lo > cross_r_hi:
            continue

        cross_r = int(rng.integers(cross_r_lo, cross_r_hi + 1))
        cross_c = int(rng.integers(cross_c_lo, cross_c_hi + 1))
        h_anchor_r = cross_r - h_local_row
        v_anchor_c = cross_c - v_local_col

        h_left = _shape_cells(h_shape, (h_anchor_r, left_h_c))
        h_right = _shape_cells(h_shape, (h_anchor_r, right_h_c))
        v_top = _shape_cells(v_shape, (top_v_r, v_anchor_c))
        v_bottom = _shape_cells(v_shape, (bottom_v_r, v_anchor_c))
        all_candidates = [h_left, h_right, v_top, v_bottom]

        temp_halo = set(occupied_halo)
        ok = True
        for cells in all_candidates:
            if not _can_place_cells(cells, temp_halo, h, w, work_r0, panel_box):
                ok = False
                break
            temp_halo |= _shape_halo(cells, h, w)
        if not ok:
            continue

        h_path = _pair_path_from_cells(h_left, h_right)
        v_path = _pair_path_from_cells(v_top, v_bottom)
        if not h_path or not v_path:
            continue
        if (cross_r, cross_c) not in h_path or (cross_r, cross_c) not in v_path:
            continue
        if (h_path & v_path) != {(cross_r, cross_c)}:
            continue
        if h_path & occupied_core or v_path & occupied_core:
            continue

        placements: dict[int, list[list[tuple[int, int]]]] = defaultdict(list)
        paths: dict[int, set[tuple[int, int]]] = defaultdict(set)
        placements[h_color] = [h_left, h_right]
        placements[v_color] = [v_top, v_bottom]
        paths[h_color] = set(h_path)
        paths[v_color] = set(v_path)
        flat_cells = [cell for cells in all_candidates for cell in cells]
        return placements, paths, _shape_halo(flat_cells, h, w), set(flat_cells)
    return None


def _paint_shapes(grid: np.ndarray, placements: dict[int, list[list[tuple[int, int]]]], color: int) -> None:
    for cells in placements.get(color, []):
        for r, c in cells:
            grid[r, c] = YELLOW


def _compose_expected_output(
    grid: np.ndarray,
    planned_shapes: dict[int, list[list[tuple[int, int]]]],
    priority: list[int],
    planned_paths: dict[int, set[tuple[int, int]]],
    panel_box: tuple[int, int, int, int],
) -> np.ndarray:
    out = grid.copy()
    for color, comps in planned_shapes.items():
        for cells in comps:
            for r, c in cells:
                out[r, c] = color
    out = _apply_priority(out, planned_paths, priority)
    r0, c0, r1, c1 = panel_box
    out[r0 : r1 + 1, c0 : c1 + 1] = grid[r0 : r1 + 1, c0 : c1 + 1]
    return out


def _generator_semantics_ok(
    grid: np.ndarray,
    legend_shapes: list[frozenset[tuple[int, int]]],
    legend_colors: list[int],
    planned_shapes: dict[int, list[list[tuple[int, int]]]],
    planned_paths: dict[int, set[tuple[int, int]]],
    panel_box: tuple[int, int, int, int],
) -> bool:
    try:
        shape_to_color, priority, detected_panel = _extract_legend(grid)
        if detected_panel != panel_box:
            return False
        if priority != legend_colors:
            return False
        for shape, color in zip(legend_shapes, legend_colors):
            if shape_to_color.get(shape) != color:
                return False

        groups = _collect_scattered_components(grid, panel_box, shape_to_color)
        base = grid.copy()
        for color, comps in groups.items():
            for comp in comps:
                rows, cols = zip(*comp["coords"])
                base[list(rows), list(cols)] = color

        actual_paths = _all_fill_paths(base, groups)
        expected_paths = {color: set(planned_paths.get(color, set())) for color in legend_colors}
        if {c: set(v) for c, v in actual_paths.items() if v} != {c: v for c, v in expected_paths.items() if v}:
            return False

        if not any(expected_paths.values()):
            return False

        total_shapes = sum(len(v) for v in groups.values())
        if not 4 <= total_shapes <= 8:
            return False
        if any(len(v) not in {2, 4} for v in groups.values()):
            return False

        expected = _compose_expected_output(grid, planned_shapes, priority, expected_paths, panel_box)
        actual = main(grid)
        if not np.array_equal(expected, actual):
            return False

        crossing_counts: dict[tuple[int, int], list[int]] = defaultdict(list)
        for color, cells in expected_paths.items():
            for cell in cells:
                crossing_counts[cell].append(color)
        crossed = [cell for cell, colors in crossing_counts.items() if len(colors) >= 2]
        if crossed:
            for cell in crossed:
                winner = min(crossing_counts[cell], key=lambda color: priority.index(color))
                if int(actual[cell]) != winner:
                    return False
        return True
    except AssertionError:
        return False


def _attempt_generate(rng: np.random.Generator, cross_mode: bool) -> np.ndarray | None:
    legend_count = 2 if cross_mode else int(rng.choice([2, 3]))
    legend_colors = list(map(int, rng.choice(np.array(LABEL_COLORS), size=legend_count, replace=False)))
    shape_indices = rng.choice(len(SHAPE_LIBRARY), size=legend_count, replace=False)
    legend_shapes = [SHAPE_LIBRARY[int(idx)] for idx in shape_indices]

    panel_h = 3 * legend_count + 1
    h = int(rng.integers(max(panel_h + 8, 16), min(panel_h + 14, 27)))
    w = int(rng.integers(16, 23))
    grid = np.zeros((h, w), dtype=int)
    panel_box = _build_panel(grid, legend_shapes, legend_colors, rng)
    work_r0 = panel_h + 1
    work_c0 = 7
    if work_r0 + 5 >= h or work_c0 + 6 >= w:
        return None

    planned_shapes: dict[int, list[list[tuple[int, int]]]] = defaultdict(list)
    planned_paths: dict[int, set[tuple[int, int]]] = defaultdict(set)
    occupied_halo: set[tuple[int, int]] = set()
    occupied_core: set[tuple[int, int]] = set()
    reserved_paths: list[set[tuple[int, int]]] = []

    use_crossing = cross_mode
    if use_crossing:
        cross_order = ["H", "V"] if bool(rng.integers(0, 2)) else ["V", "H"]
        cross_specs = [
            (legend_colors[0], legend_shapes[0], cross_order[0]),
            (legend_colors[1], legend_shapes[1], cross_order[1]),
        ]
        placed = _place_crossing_pairs(
            cross_specs,
            occupied_halo,
            occupied_core,
            h,
            w,
            work_r0,
            work_c0,
            panel_box,
            rng,
        )
        if placed is None:
            return None
        cross_shapes, cross_paths, added_halo, added_core = placed
        for color, comps in cross_shapes.items():
            planned_shapes[color].extend(comps)
        for color, cells in cross_paths.items():
            planned_paths[color] |= cells
            reserved_paths.append(set(cells))
        occupied_halo |= added_halo
        occupied_core |= added_core
        start_idx = 2
    else:
        start_idx = 0

    for idx in range(start_idx, legend_count):
        orientation = "H" if bool(rng.integers(0, 2)) else "V"
        placed = _place_pair(
            legend_shapes[idx],
            orientation,
            occupied_halo,
            occupied_core,
            reserved_paths,
            h,
            w,
            work_r0,
            work_c0,
            panel_box,
            rng,
        )
        if placed is None:
            return None
        left_cells, right_cells, path = placed
        planned_shapes[legend_colors[idx]].extend([left_cells, right_cells])
        planned_paths[legend_colors[idx]] |= path
        occupied_halo |= _shape_halo(left_cells, h, w) | _shape_halo(right_cells, h, w)
        occupied_core |= set(left_cells) | set(right_cells)
        reserved_paths.append(set(path))

    extra_color = None
    if sum(len(v) for v in planned_shapes.values()) <= 6 and bool(rng.integers(0, 2)):
        extra_idx = int(rng.integers(legend_count))
        extra_color = legend_colors[extra_idx]
        extra_shape = legend_shapes[extra_idx]
        for _ in range(300):
            a_r = int(rng.integers(work_r0, h - 1))
            a_c = int(rng.integers(work_c0, w - 1))
            b_r = int(rng.integers(work_r0, h - 1))
            b_c = int(rng.integers(work_c0, w - 1))
            if abs(a_r - b_r) <= 1 or abs(a_c - b_c) <= 1:
                continue
            extra_a = _shape_cells(extra_shape, (a_r, a_c))
            if not _can_place_cells(extra_a, occupied_halo, h, w, work_r0, panel_box):
                continue
            extra_b = _shape_cells(extra_shape, (b_r, b_c))
            if not _can_place_cells(extra_b, occupied_halo | _shape_halo(extra_a, h, w), h, w, work_r0, panel_box):
                continue

            same_color_existing = [_cells_to_profile(cells) for cells in planned_shapes[extra_color]]
            extra_profiles = [_cells_to_profile(extra_a), _cells_to_profile(extra_b)]
            dummy = np.zeros((h, w), dtype=int)
            accidental = False
            for prof in same_color_existing:
                if _pair_fill_cells(prof, extra_profiles[0], dummy) or _pair_fill_cells(prof, extra_profiles[1], dummy):
                    accidental = True
                    break
            if accidental or _pair_fill_cells(extra_profiles[0], extra_profiles[1], dummy):
                continue

            planned_shapes[extra_color].extend([extra_a, extra_b])
            occupied_halo |= _shape_halo(extra_a, h, w) | _shape_halo(extra_b, h, w)
            occupied_core |= set(extra_a) | set(extra_b)
            break
        else:
            return None

    total_shapes = sum(len(v) for v in planned_shapes.values())
    if not 4 <= total_shapes <= 8:
        return None
    if any(len(v) not in {2, 4} for v in planned_shapes.values()):
        return None

    for color in legend_colors:
        for cells in planned_shapes[color]:
            for r, c in cells:
                grid[r, c] = YELLOW

    if extra_color is None and not use_crossing and legend_count == 2:
        # Ensure at least one negative branch appears often enough by forcing a 4-shape color in some 2-color cases.
        return None

    if not _generator_semantics_ok(
        grid,
        legend_shapes,
        legend_colors,
        planned_shapes,
        planned_paths,
        panel_box,
    ):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    first_mode = bool(rng.integers(0, 2))
    for mode in (first_mode, not first_mode):
        for _ in range(300):
            grid = _attempt_generate(rng, mode)
            if grid is not None:
                return grid
    raise RuntimeError("failed to generate a valid input")
