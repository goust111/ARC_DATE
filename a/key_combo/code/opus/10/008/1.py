from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
GREY = 5
EMITTER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
RECOLOR_BY_COUNT = {2: 3, 3: 2, 4: 1, 5: 4}
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NEIGHBORS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]
RNG = np.random.default_rng()


def main(grid: np.ndarray) -> np.ndarray:
    """Extend singleton emitters into row/column rays, then recolor each ray-bounded region by its grey-pixel total.

    Input structure:
    - A black background contains 3-5 isolated non-grey singleton emitters and multiple grey 4-connected mini-objects.
    - Emitters occupy distinct rows and distinct columns; grey pixels never lie on an emitter row or emitter column.
    - The emitter rows and columns, together with the outer border, define rectangular counting regions.

    Rule:
    - Copy the grid.
    - From every emitter, paint its full row and full column with that emitter color, but only onto background cells; non-background cells are preserved.
    - For each rectangle between consecutive emitter rows/columns (including the edge bands), count grey pixels inside that rectangle and recolor only those grey pixels: 2->green, 3->red, 4->blue, 5->yellow, anything else stays grey.

    Key insight:
    - The colored singletons are not labels for nearby objects; they define a global row/column partition, and each region's combined grey-pixel count determines its recolor.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    emitters = _find_emitters(g)
    assert 3 <= len(emitters) <= 5, "expected 3-5 emitters"

    row_lines = sorted({r for r, _, _ in emitters})
    col_lines = sorted({c for _, c, _ in emitters})
    assert len(row_lines) == len(emitters), "emitters must have unique rows"
    assert len(col_lines) == len(emitters), "emitters must have unique columns"

    out = _paint_rays(g, emitters, row_lines, col_lines)
    row_spans = _region_spans(g.shape[0], row_lines)
    col_spans = _region_spans(g.shape[1], col_lines)

    for r0, r1 in row_spans:
        if r0 > r1:
            continue
        for c0, c1 in col_spans:
            if c0 > c1:
                continue
            region = g[r0 : r1 + 1, c0 : c1 + 1]
            grey_count = int(np.count_nonzero(region == GREY))
            recolor = RECOLOR_BY_COUNT.get(grey_count, GREY)
            if recolor != GREY:
                mask = out[r0 : r1 + 1, c0 : c1 + 1] == GREY
                out[r0 : r1 + 1, c0 : c1 + 1][mask] = recolor

    return out


def _find_emitters(grid: np.ndarray) -> list[tuple[int, int, int]]:
    coords = np.argwhere((grid != BG) & (grid != GREY))
    emitters = [(int(r), int(c), int(grid[r, c])) for r, c in coords]
    emitters.sort()
    return emitters


def _paint_rays(
    grid: np.ndarray,
    emitters: list[tuple[int, int, int]],
    row_lines: list[int],
    col_lines: list[int],
) -> np.ndarray:
    out = grid.copy()
    emitter_positions = {(r, c): color for r, c, color in emitters}
    crossing_cells = {(r, c) for r in row_lines for c in col_lines}

    for r, c, color in emitters:
        for cc in range(grid.shape[1]):
            if (r, cc) in crossing_cells and (r, cc) not in emitter_positions:
                continue
            if out[r, cc] == BG:
                out[r, cc] = color
        for rr in range(grid.shape[0]):
            if (rr, c) in crossing_cells and (rr, c) not in emitter_positions:
                continue
            if out[rr, c] == BG:
                out[rr, c] = color

    for (r, c), color in emitter_positions.items():
        out[r, c] = color
    for r, c in crossing_cells:
        if (r, c) not in emitter_positions:
            out[r, c] = BG
    return out


def _region_spans(length: int, lines: list[int]) -> list[tuple[int, int]]:
    spans = []
    start = 0
    for line in lines:
        spans.append((start, line - 1))
        start = line + 1
    spans.append((start, length - 1))
    return spans


def _spans_from_intervals(intervals: list[int]) -> list[tuple[int, int]]:
    spans = []
    start = 0
    for size in intervals:
        spans.append((start, start + size - 1))
        start += size + 1
    return spans


def _lines_from_intervals(intervals: list[int]) -> list[int]:
    lines = []
    cursor = intervals[0]
    lines.append(cursor)
    for size in intervals[1:-1]:
        cursor += 1 + size
        lines.append(cursor)
    return lines


def _sample_intervals(emitter_count: int) -> list[int] | None:
    band_count = emitter_count + 1
    max_size = {3: 6, 4: 5, 5: 4}[emitter_count]
    for _ in range(200):
        intervals = RNG.integers(3, max_size + 1, size=band_count).tolist()
        if max(intervals) < 4:
            continue
        return [int(x) for x in intervals]
    return None


def _sample_region_components(height: int, width: int) -> list[set[tuple[int, int]]] | None:
    order = RNG.permutation(len(PLAN_CATALOG))
    weights = PLAN_WEIGHTS[order]
    weights = weights / weights.sum()
    for choice in RNG.choice(order, size=len(order), replace=False, p=weights):
        plan = PLAN_CATALOG[int(choice)]
        components = _place_plan_in_region(height, width, plan)
        if components is not None:
            return components
    return None


def _place_plan_in_region(
    height: int,
    width: int,
    plan: tuple[int, ...],
) -> list[set[tuple[int, int]]] | None:
    sizes = sorted((int(size) for size in plan), reverse=True)
    occupied: set[tuple[int, int]] = set()
    forbidden: set[tuple[int, int]] = set()

    def search(index: int) -> list[set[tuple[int, int]]] | None:
        if index == len(sizes):
            return []
        size = sizes[index]
        candidates = []
        for shape in ORIENTED_SHAPES[size]:
            shape_h = 1 + max(r for r, _ in shape)
            shape_w = 1 + max(c for _, c in shape)
            if shape_h > height or shape_w > width:
                continue
            for r0 in range(height - shape_h + 1):
                for c0 in range(width - shape_w + 1):
                    cells = {(r0 + r, c0 + c) for r, c in shape}
                    if cells & forbidden:
                        continue
                    candidates.append(cells)
        if not candidates:
            return None

        RNG.shuffle(candidates)
        for cells in candidates:
            added_forbidden = set()
            for r, c in cells:
                occupied.add((r, c))
                for dr, dc in NEIGHBORS8 + [(0, 0)]:
                    rr = r + dr
                    cc = c + dc
                    if 0 <= rr < height and 0 <= cc < width:
                        if (rr, cc) not in forbidden:
                            added_forbidden.add((rr, cc))
                        forbidden.add((rr, cc))
            tail = search(index + 1)
            if tail is not None:
                return [cells] + tail
            for cell in cells:
                occupied.remove(cell)
            forbidden.difference_update(added_forbidden)
        return None

    return search(0)


def _all_orientations(shapes: list[tuple[tuple[int, int], ...]]) -> list[tuple[tuple[int, int], ...]]:
    out = set()
    for shape in shapes:
        arr = np.array(shape, dtype=int)
        for flip in (1, -1):
            pts = arr.copy()
            pts[:, 1] *= flip
            for _ in range(4):
                normalized = _normalize_shape(pts)
                out.add(tuple(sorted((int(r), int(c)) for r, c in normalized)))
                pts = np.column_stack([pts[:, 1], -pts[:, 0]])
    return sorted(out)


def _normalize_shape(coords: np.ndarray) -> np.ndarray:
    shifted = coords.copy()
    shifted[:, 0] -= int(shifted[:, 0].min())
    shifted[:, 1] -= int(shifted[:, 1].min())
    return shifted


BASE_SHAPES = {
    2: [((0, 0), (0, 1))],
    3: [((0, 0), (0, 1), (0, 2)), ((0, 0), (1, 0), (1, 1))],
    4: [
        ((0, 0), (0, 1), (0, 2), (0, 3)),
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        ((0, 0), (1, 0), (2, 0), (2, 1)),
        ((0, 1), (1, 0), (1, 1), (1, 2)),
    ],
    5: [
        ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
        ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)),
        ((0, 1), (1, 1), (2, 0), (2, 1), (2, 2)),
    ],
}
ORIENTED_SHAPES = {area: _all_orientations(shapes) for area, shapes in BASE_SHAPES.items()}
PLAN_CATALOG = [
    (2,),
    (3,),
    (4,),
    (5,),
    (2, 2),
    (2, 3),
    (2, 4),
    (3, 3),
]
SPECIAL_PLANS = [
    (2,),
    (3,),
    (4,),
    (5,),
    (2, 2),
    (2, 3),
    (2, 4),
]
PLAN_WEIGHTS = np.array([1.2, 1.2, 1.0, 1.0, 1.0, 1.0, 0.7, 0.6], dtype=float)


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _generate_attempt()
        if grid is not None:
            _assert_valid_instance(grid)
            return grid
    raise RuntimeError("failed to generate valid ray-partition recolor input")


def _generate_attempt() -> np.ndarray | None:
    emitter_count = int(RNG.choice([3, 4, 5], p=[0.48, 0.34, 0.18]))
    row_intervals = _sample_intervals(emitter_count)
    col_intervals = _sample_intervals(emitter_count)
    if row_intervals is None or col_intervals is None:
        return None

    height = sum(row_intervals) + emitter_count
    width = sum(col_intervals) + emitter_count
    if height > 30 or width > 30:
        return None

    row_lines = _lines_from_intervals(row_intervals)
    col_lines = _lines_from_intervals(col_intervals)
    row_spans = _spans_from_intervals(row_intervals)
    col_spans = _spans_from_intervals(col_intervals)

    colors = RNG.choice(np.array(EMITTER_COLORS, dtype=int), size=emitter_count, replace=False).tolist()
    perm = RNG.permutation(emitter_count)
    emitters = [(row_lines[i], col_lines[int(perm[i])], int(colors[i])) for i in range(emitter_count)]

    grid = np.zeros((height, width), dtype=int)
    for r, c, color in emitters:
        grid[r, c] = color

    region_boxes = []
    for rr, (r0, r1) in enumerate(row_spans):
        for cc, (c0, c1) in enumerate(col_spans):
            region_boxes.append(
                {
                    "index": (rr, cc),
                    "bbox": (r0, r1, c0, c1),
                    "height": r1 - r0 + 1,
                    "width": c1 - c0 + 1,
                    "area": (r1 - r0 + 1) * (c1 - c0 + 1),
                }
            )

    anchors = sorted(region_boxes, key=lambda region: (region["area"], region["height"], region["width"]), reverse=True)
    placements_by_region: dict[tuple[int, int], list[set[tuple[int, int]]]] = {}
    used_indices: set[tuple[int, int]] = set()
    for plan in SPECIAL_PLANS:
        placed = False
        for region in anchors:
            idx = region["index"]
            if idx in used_indices:
                continue
            components = _place_plan_in_region(region["height"], region["width"], plan)
            if components is None:
                continue
            placements_by_region[idx] = [
                {(region["bbox"][0] + r, region["bbox"][2] + c) for r, c in component}
                for component in components
            ]
            used_indices.add(idx)
            placed = True
            break
        if not placed:
            return None

    for region in region_boxes:
        idx = region["index"]
        if idx in placements_by_region:
            continue
        components = _sample_region_components(region["height"], region["width"])
        if components is None:
            return None
        placements_by_region[idx] = [
            {(region["bbox"][0] + r, region["bbox"][2] + c) for r, c in component}
            for component in components
        ]

    for components in placements_by_region.values():
        for cells in components:
            for r, c in cells:
                grid[r, c] = GREY

    if np.any(grid[row_lines, :] == GREY) or np.any(grid[:, col_lines] == GREY):
        return None

    counts = _region_grey_counts(grid, row_lines, col_lines)
    if not {2, 3, 4, 5}.issubset(set(counts.values())):
        return None
    if not any(total > 5 for total in counts.values()):
        return None

    components = _grey_components(grid)
    region_component_counts = _region_component_counts(components, row_lines, col_lines)
    if max(region_component_counts.values()) < 2:
        return None

    expected = _expected_output_from_rules(grid)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    return grid


def _grey_components(grid: np.ndarray) -> list[dict]:
    visited = np.zeros_like(grid, dtype=bool)
    components = []
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if visited[r, c] or grid[r, c] != GREY:
                continue
            queue = deque([(r, c)])
            visited[r, c] = True
            coords = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and not visited[nr, nc] and grid[nr, nc] == GREY:
                        visited[nr, nc] = True
                        queue.append((nr, nc))
            components.append({"coords": coords, "area": len(coords)})
    return components


def _component_id_grid(grid: np.ndarray) -> np.ndarray:
    comp_ids = np.full(grid.shape, -1, dtype=int)
    for cid, comp in enumerate(_grey_components(grid)):
        for r, c in comp["coords"]:
            comp_ids[r, c] = cid
    return comp_ids


def _region_grey_counts(grid: np.ndarray, row_lines: list[int], col_lines: list[int]) -> dict[tuple[int, int], int]:
    counts = {}
    for rr, (r0, r1) in enumerate(_region_spans(grid.shape[0], row_lines)):
        if r0 > r1:
            continue
        for cc, (c0, c1) in enumerate(_region_spans(grid.shape[1], col_lines)):
            if c0 > c1:
                continue
            counts[(rr, cc)] = int(np.count_nonzero(grid[r0 : r1 + 1, c0 : c1 + 1] == GREY))
    return counts


def _region_component_counts(
    components: list[dict],
    row_lines: list[int],
    col_lines: list[int],
) -> dict[tuple[int, int], int]:
    counts = {key: 0 for key in _region_spans_map(len(row_lines) + 1, len(col_lines) + 1)}
    for comp in components:
        r, c = comp["coords"][0]
        region = (_region_index_from_lines(r, row_lines), _region_index_from_lines(c, col_lines))
        counts[region] += 1
    return counts


def _region_spans_map(row_regions: int, col_regions: int) -> dict[tuple[int, int], int]:
    return {(rr, cc): 0 for rr in range(row_regions) for cc in range(col_regions)}


def _region_index_from_lines(value: int, lines: list[int]) -> int:
    idx = 0
    while idx < len(lines) and value > lines[idx]:
        idx += 1
    return idx


def _expected_output_from_rules(grid: np.ndarray) -> np.ndarray:
    emitters = _find_emitters(grid)
    row_lines = sorted({r for r, _, _ in emitters})
    col_lines = sorted({c for _, c, _ in emitters})
    out = _paint_rays(grid, emitters, row_lines, col_lines)
    for (rr, cc), grey_count in _region_grey_counts(grid, row_lines, col_lines).items():
        recolor = RECOLOR_BY_COUNT.get(grey_count, GREY)
        if recolor == GREY:
            continue
        r0, r1 = _region_spans(grid.shape[0], row_lines)[rr]
        c0, c1 = _region_spans(grid.shape[1], col_lines)[cc]
        mask = out[r0 : r1 + 1, c0 : c1 + 1] == GREY
        out[r0 : r1 + 1, c0 : c1 + 1][mask] = recolor
    return out


def _assert_valid_instance(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2
    assert max(grid.shape) <= 30

    emitters = _find_emitters(grid)
    assert 3 <= len(emitters) <= 5
    assert len({color for _, _, color in emitters}) == len(emitters)
    row_lines = sorted({r for r, _, _ in emitters})
    col_lines = sorted({c for _, c, _ in emitters})
    assert len(row_lines) == len(emitters)
    assert len(col_lines) == len(emitters)
    assert not np.any(grid[row_lines, :] == GREY)
    assert not np.any(grid[:, col_lines] == GREY)

    components = _grey_components(grid)
    assert components
    assert all(2 <= comp["area"] <= 5 for comp in components)

    comp_ids = _component_id_grid(grid)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if comp_ids[r, c] < 0:
                continue
            for dr, dc in NEIGHBORS8:
                rr = r + dr
                cc = c + dc
                if 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1] and comp_ids[rr, cc] >= 0:
                    assert comp_ids[rr, cc] == comp_ids[r, c], "grey components must not 8-touch"

    counts = _region_grey_counts(grid, row_lines, col_lines)
    assert counts
    assert min(counts.values()) >= 2
    assert {2, 3, 4, 5}.issubset(set(counts.values()))
    assert any(total > 5 for total in counts.values())

    region_component_counts = _region_component_counts(components, row_lines, col_lines)
    assert max(region_component_counts.values()) >= 2

    out = main(grid)
    assert np.any(out == 1) and np.any(out == 2) and np.any(out == 3) and np.any(out == 4)
    for (rr, cc), total in counts.items():
        r0, r1 = _region_spans(grid.shape[0], row_lines)[rr]
        c0, c1 = _region_spans(grid.shape[1], col_lines)[cc]
        region_out = out[r0 : r1 + 1, c0 : c1 + 1]
        region_in = grid[r0 : r1 + 1, c0 : c1 + 1]
        if total in RECOLOR_BY_COUNT:
            assert np.all(region_out[region_in == GREY] == RECOLOR_BY_COUNT[total])
        else:
            assert np.all(region_out[region_in == GREY] == GREY)
