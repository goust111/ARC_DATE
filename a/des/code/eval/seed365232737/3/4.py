import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
CYAN = 8
MAROON = 9

SOURCE_COLORS = (RED, GREEN, YELLOW, MAGENTA, CYAN, MAROON)

GEN_RETRIES = 300
MIN_REGION_WIDTH = 2
MIN_BAND_STEP = 2


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _label_components(mask: np.ndarray):
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    comps: list[list[tuple[int, int]]] = []
    label = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            label += 1
            q = deque([(r, c)])
            labels[r, c] = label
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = label
                        q.append((nr, nc))
            comps.append(cells)
    return labels, comps


def _analyze_grid(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    barrier = (g == BLUE) | (g == ORANGE)
    region_labels, region_cells = _label_components(~barrier)
    assert region_cells, "expected at least one region"

    region_sources: list[set[int]] = []
    for cells in region_cells:
        colors = {
            int(g[r, c])
            for r, c in cells
            if int(g[r, c]) not in (BLACK, BLUE, ORANGE)
        }
        region_sources.append(colors)

    _, bridge_cells = _label_components(g == ORANGE)
    edges: set[tuple[int, int]] = set()
    h, w = g.shape
    for comp in bridge_cells:
        adjacent_regions: set[int] = set()
        for r, c in comp:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and region_labels[nr, nc] != 0:
                    adjacent_regions.add(int(region_labels[nr, nc]) - 1)
        assert len(adjacent_regions) == 2, "each bridge must connect exactly two regions"
        a, b = sorted(adjacent_regions)
        edges.add((a, b))

    return region_labels, region_cells, region_sources, edges


def _graph_components(n_nodes: int, edges: set[tuple[int, int]]):
    adj = [set() for _ in range(n_nodes)]
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)

    comp_of = [-1] * n_nodes
    comps: list[list[int]] = []
    for start in range(n_nodes):
        if comp_of[start] != -1:
            continue
        comp_id = len(comps)
        q = deque([start])
        comp_of[start] = comp_id
        nodes: list[int] = []
        while q:
            node = q.popleft()
            nodes.append(node)
            for nxt in adj[node]:
                if comp_of[nxt] == -1:
                    comp_of[nxt] = comp_id
                    q.append(nxt)
        comps.append(nodes)
    return comp_of, comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill river-separated regions from source pixels, propagate through bridges,
    and paint multi-source bridge components grey.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    _region_labels, region_cells, region_sources, edges = _analyze_grid(g)
    comp_of, comps = _graph_components(len(region_cells), edges)

    comp_source_colors: list[set[int]] = []
    for comp in comps:
        colors: set[int] = set()
        for rid in comp:
            colors |= region_sources[rid]
        comp_source_colors.append(colors)

    out = g.copy()
    for rid, cells in enumerate(region_cells):
        own_sources = region_sources[rid]
        if len(own_sources) == 1:
            fill_color = next(iter(own_sources))
        elif len(own_sources) > 1:
            fill_color = GREY
        else:
            colors = comp_source_colors[comp_of[rid]]
            if len(colors) == 1:
                fill_color = next(iter(colors))
            elif len(colors) > 1:
                fill_color = GREY
            else:
                fill_color = None

        if fill_color is None:
            continue
        for r, c in cells:
            if out[r, c] == BLACK:
                out[r, c] = fill_color

    return out


# --------------- generator ---------------


def _random_partition(total: int, parts: int, minimum: int) -> list[int]:
    values = [minimum] * parts
    for _ in range(total - parts * minimum):
        values[random.randrange(parts)] += 1
    random.shuffle(values)
    return values


def _river_columns(widths: list[int]) -> list[int]:
    cols = [widths[0]]
    for width in widths[1:-1]:
        cols.append(cols[-1] + 1 + width)
    return cols


def _mutate_widths(widths: list[int]) -> list[int]:
    for _ in range(20):
        new = widths[:]
        changed = False
        for _ in range(random.randint(1, min(3, len(widths) - 1))):
            boundary = random.randrange(len(widths) - 1)
            if random.random() < 0.5:
                donor, receiver = boundary, boundary + 1
            else:
                donor, receiver = boundary + 1, boundary
            if new[donor] <= MIN_REGION_WIDTH:
                continue
            new[donor] -= 1
            new[receiver] += 1
            changed = True
        if changed and new != widths:
            return new
    return widths[:]


def _ordered_regions_from_top(grid: np.ndarray):
    labels, comps = _label_components(grid == BLACK)
    ordered: list[int] = []
    w = grid.shape[1]
    c = 0
    while c < w:
        if grid[0, c] != BLACK:
            c += 1
            continue
        label = int(labels[0, c]) - 1
        ordered.append(label)
        while c < w and grid[0, c] == BLACK:
            c += 1
    assert len(ordered) == len(comps), "top row should expose every strip region"
    return labels, comps, ordered


def _bridge_options(segments: list[tuple[int, int, int]], length: int):
    options: list[tuple[int, int, int]] = []
    for col, start_row, end_row in segments:
        lo = start_row + 1
        hi = end_row - length
        for r0 in range(lo, hi + 1):
            options.append((col, r0, length))
    return options


def _preferred_source_cells(grid: np.ndarray, cells: list[tuple[int, int]]):
    h, w = grid.shape
    preferred: list[tuple[int, int]] = []
    interior: list[tuple[int, int]] = []
    for r, c in cells:
        if grid[r, c] != BLACK:
            continue
        if 0 < r < h - 1 and 0 < c < w - 1:
            interior.append((r, c))
            if all(
                0 <= nr < h and 0 <= nc < w and grid[nr, nc] == BLACK
                for nr, nc in _neighbors4(r, c)
            ):
                preferred.append((r, c))
    if preferred:
        return preferred
    if interior:
        return interior
    return [cell for cell in cells if grid[cell] == BLACK]


def _plan_single(n_regions: int):
    length = random.randint(2, n_regions)
    start = random.randint(0, n_regions - length)
    active_edges = set(range(start, start + length - 1))
    source_regions = [random.randint(start, start + length - 1)]
    return active_edges, source_regions


def _plan_conflict(n_regions: int):
    length = random.randint(3, n_regions)
    start = random.randint(0, n_regions - length)
    active_edges = set(range(start, start + length - 1))
    source_regions = [start, start + length - 1]
    return active_edges, source_regions


def _plan_mixed(n_regions: int):
    candidate_pairs = [
        (a, b)
        for a in range(n_regions - 1)
        for b in range(a + 2, n_regions - 1)
    ]
    if candidate_pairs and random.random() < 0.5:
        edge_a, edge_b = random.choice(candidate_pairs)
        active_edges = {edge_a, edge_b}
        source_regions = [random.choice((edge_a, edge_a + 1)), random.choice((edge_b, edge_b + 1))]
        return active_edges, source_regions

    length = random.randint(2, n_regions - 1)
    start = random.randint(0, n_regions - length)
    component = list(range(start, start + length))
    outside = [idx for idx in range(n_regions) if idx not in component]
    active_edges = set(range(start, start + length - 1))
    source_regions = [random.choice(component), random.choice(outside)]
    return active_edges, source_regions


def _apply_random_symmetry(grid: np.ndarray) -> np.ndarray:
    out = np.rot90(grid, random.randrange(4)).copy()
    if random.random() < 0.5:
        out = np.fliplr(out).copy()
    return out


def _build_vertical_layout():
    n_regions = random.randint(3, 5)
    total_region_width = n_regions * MIN_REGION_WIDTH + random.randint(max(2, n_regions - 1), n_regions + 4)
    widths0 = _random_partition(total_region_width, n_regions, MIN_REGION_WIDTH)
    grid_w = total_region_width + n_regions - 1

    grid_h = random.randint(13, 20)
    max_bands = min(5, (grid_h - 1) // MIN_BAND_STEP)
    n_bands = random.randint(3, max_bands)
    band_lengths = _random_partition(grid_h - 1, n_bands, MIN_BAND_STEP)

    widths_by_band = [widths0]
    changed_any = False
    for _ in range(n_bands - 1):
        nxt = _mutate_widths(widths_by_band[-1])
        changed_any |= nxt != widths_by_band[-1]
        widths_by_band.append(nxt)
    if not changed_any:
        widths_by_band[1] = _mutate_widths(widths_by_band[0])

    cols_by_band = [_river_columns(widths) for widths in widths_by_band]
    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    segments: list[list[tuple[int, int, int]]] = [[] for _ in range(n_regions - 1)]
    end_rows = np.cumsum(band_lengths)

    for river_idx in range(n_regions - 1):
        start_row = 0
        cur_col = cols_by_band[0][river_idx]
        for band_idx, end_row_raw in enumerate(end_rows):
            end_row = int(end_row_raw)
            segments[river_idx].append((cur_col, start_row, end_row))
            grid[start_row : end_row + 1, cur_col] = BLUE
            if band_idx + 1 < n_bands:
                next_col = cols_by_band[band_idx + 1][river_idx]
                grid[end_row, min(cur_col, next_col) : max(cur_col, next_col) + 1] = BLUE
                start_row = end_row
                cur_col = next_col

    return grid, segments, n_regions


def _gen_attempt() -> np.ndarray | None:
    scenario = random.choice(("single", "mixed", "conflict"))
    grid, segments, n_regions = _build_vertical_layout()

    if scenario == "single":
        active_edges, source_regions = _plan_single(n_regions)
    elif scenario == "mixed":
        active_edges, source_regions = _plan_mixed(n_regions)
    else:
        active_edges, source_regions = _plan_conflict(n_regions)

    for river_idx in active_edges:
        length = 2 if random.random() < 0.5 else 1
        options = _bridge_options(segments[river_idx], length)
        if not options:
            options = _bridge_options(segments[river_idx], 1)
        if not options:
            return None
        col, start_row, used_length = random.choice(options)
        grid[start_row : start_row + used_length, col] = ORANGE

    _labels, region_cells, ordered = _ordered_regions_from_top(grid)
    if len(ordered) != n_regions:
        return None

    source_regions = list(dict.fromkeys(source_regions))
    colors = random.sample(SOURCE_COLORS, len(source_regions))
    for region_idx, color in zip(source_regions, colors):
        actual_region = ordered[region_idx]
        candidates = _preferred_source_cells(grid, region_cells[actual_region])
        if not candidates:
            return None
        r, c = random.choice(candidates)
        grid[r, c] = color

    analyzed = _analyze_grid(grid)
    if len(analyzed[1]) != n_regions or len(analyzed[3]) != len(active_edges):
        return None

    grid = _apply_random_symmetry(grid)
    out = main(grid)
    if np.array_equal(grid, out):
        return None
    if scenario == "conflict" and not np.any(out == GREY):
        return None
    if scenario != "conflict" and np.any(out == GREY):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
