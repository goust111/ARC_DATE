from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


Coord = tuple[int, int]
Edge = tuple[Coord, Coord]

GEN_ATTEMPTS = 400
ROW_COUNT_RANGE = (4, 5)
COL_COUNT_RANGE = (4, 5)
CELL_SIZE_RANGE = (2, 5)
EXTRA_BRANCH_RANGE = (2, 8)


def _norm_edge(a: Coord, b: Coord) -> Edge:
    return (a, b) if a <= b else (b, a)


def _neighbors4(coord: Coord, rows: int, cols: int):
    r, c = coord
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr = r + dr
        nc = c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            yield nr, nc


def _line_path(a: Coord, b: Coord) -> list[Coord]:
    ar, ac = a
    br, bc = b
    if ar == br:
        step = 1 if bc >= ac else -1
        return [(ar, c) for c in range(ac, bc + step, step)]
    if ac == bc:
        step = 1 if br >= ar else -1
        return [(r, ac) for r in range(ar, br + step, step)]
    raise ValueError("orthogonal segment required")


def _stitch_segments(points: list[Coord]) -> list[Coord]:
    path: list[Coord] = []
    for i in range(len(points) - 1):
        segment = _line_path(points[i], points[i + 1])
        if path:
            segment = segment[1:]
        path.extend(segment)
    return path


def _all_lattice_edges(rows: int, cols: int) -> list[Edge]:
    edges: list[Edge] = []
    for r in range(rows):
        for c in range(cols):
            if r + 1 < rows:
                edges.append(_norm_edge((r, c), (r + 1, c)))
            if c + 1 < cols:
                edges.append(_norm_edge((r, c), (r, c + 1)))
    return edges


def _bfs_path(rows: int, cols: int, start: Coord, end: Coord, banned: set[Edge]) -> list[Coord] | None:
    queue = deque([start])
    parent = {start: None}
    while queue:
        cur = queue.popleft()
        if cur == end:
            break
        for nxt in _neighbors4(cur, rows, cols):
            edge = _norm_edge(cur, nxt)
            if edge in banned or nxt in parent:
                continue
            parent[nxt] = cur
            queue.append(nxt)
    if end not in parent:
        return None
    path: list[Coord] = []
    cur: Coord | None = end
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def _component_is_tree(adj: dict[int, set[int]], start: int, end: int) -> bool:
    stack = [start]
    seen = set()
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(adj[node] - seen)
    if end not in seen:
        return False
    edge_count = sum(len(adj[node]) for node in seen) // 2
    return edge_count == len(seen) - 1


def _find_separator_color(grid: np.ndarray) -> tuple[int, list[int], list[int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    colors = [int(v) for v in np.unique(g)]
    best: tuple[int, int, int, int] | None = None
    best_color = colors[0]
    best_rows: list[int] = []
    best_cols: list[int] = []

    for color in colors:
        seed_rows = [r for r in range(h) if int(np.count_nonzero(g[r] == color)) >= (w + 1) // 2]
        seed_cols = [c for c in range(w) if int(np.count_nonzero(g[:, c] == color)) >= (h + 1) // 2]
        if not seed_rows or not seed_cols:
            continue
        intersections = sum(int(g[r, c] == color) for r in seed_rows for c in seed_cols)
        score = (len(seed_rows) + len(seed_cols), intersections, int(np.count_nonzero(g == color)), -color)
        if best is None or score > best:
            best = score
            best_color = color
            best_rows = seed_rows
            best_cols = seed_cols

    assert best is not None, "separator color not found"
    row_slack = len(best_cols) + 1
    col_slack = len(best_rows) + 1
    sep_rows = [r for r in range(h) if int(np.count_nonzero(g[r] == best_color)) >= w - row_slack]
    sep_cols = [c for c in range(w) if int(np.count_nonzero(g[:, c] == best_color)) >= h - col_slack]
    assert sep_rows and sep_cols, "separator axes not found"
    assert all(b - a >= 2 for a, b in zip(sep_rows, sep_rows[1:])), "separator rows must be one pixel thick"
    assert all(b - a >= 2 for a, b in zip(sep_cols, sep_cols[1:])), "separator cols must be one pixel thick"
    for r in sep_rows:
        for c in sep_cols:
            assert int(g[r, c]) == best_color, "separator intersections must stay solid"
    return best_color, sep_rows, sep_cols


def _partition_ranges(length: int, bars: list[int]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start = 0
    for bar in bars:
        assert start <= bar - 1, "empty region band"
        ranges.append((start, bar - 1))
        start = bar + 1
    assert start <= length - 1, "missing trailing region band"
    ranges.append((start, length - 1))
    return ranges


def _extract_regions(grid: np.ndarray):
    sep_color, sep_rows, sep_cols = _find_separator_color(grid)
    row_ranges = _partition_ranges(grid.shape[0], sep_rows)
    col_ranges = _partition_ranges(grid.shape[1], sep_cols)
    regions: list[dict[str, object]] = []
    index_map: dict[Coord, int] = {}
    for i, (r0, r1) in enumerate(row_ranges):
        for j, (c0, c1) in enumerate(col_ranges):
            block = grid[r0 : r1 + 1, c0 : c1 + 1]
            colors = np.unique(block)
            assert len(colors) == 1, "each region must be uniform"
            idx = len(regions)
            index_map[(i, j)] = idx
            regions.append(
                {
                    "index": idx,
                    "cell": (i, j),
                    "bbox": (r0, r1, c0, c1),
                    "color": int(colors[0]),
                    "edge": i == 0 or j == 0 or i == len(row_ranges) - 1 or j == len(col_ranges) - 1,
                }
            )
    return sep_color, sep_rows, sep_cols, row_ranges, col_ranges, regions, index_map


def _same_color_chain(regions: list[dict[str, object]], shape: tuple[int, int], start: int, end: int) -> bool:
    rows, cols = shape
    color = int(regions[start]["color"])
    allowed = {int(region["index"]) for region in regions if int(region["color"]) == color}
    queue = deque([start])
    seen = {start}
    while queue:
        idx = queue.popleft()
        if idx == end:
            return True
        r, c = regions[idx]["cell"]
        for nr, nc in _neighbors4((r, c), rows, cols):
            nxt = nr * cols + nc
            if nxt in allowed and nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return False


def _marker_region_indices(regions: list[dict[str, object]], shape: tuple[int, int]) -> tuple[int, int, int]:
    by_color: dict[int, list[int]] = {}
    edge_by_color: dict[int, list[int]] = {}
    for region in regions:
        idx = int(region["index"])
        color = int(region["color"])
        by_color.setdefault(color, []).append(idx)
        if bool(region["edge"]):
            edge_by_color.setdefault(color, []).append(idx)

    candidates = []
    for color, edge_cells in edge_by_color.items():
        if len(edge_cells) != 2:
            continue
        if _same_color_chain(regions, shape, edge_cells[0], edge_cells[1]):
            candidates.append((color, tuple(edge_cells)))
    assert len(candidates) == 1, "expected exactly one marker color"
    color, pair = candidates[0]
    return int(color), int(pair[0]), int(pair[1])


def _gap_graph(
    grid: np.ndarray,
    sep_color: int,
    sep_rows: list[int],
    sep_cols: list[int],
    row_ranges: list[tuple[int, int]],
    col_ranges: list[tuple[int, int]],
    index_map: dict[Coord, int],
):
    total = len(row_ranges) * len(col_ranges)
    adj = {idx: set() for idx in range(total)}
    for i, sr in enumerate(sep_rows):
        for j, (c0, c1) in enumerate(col_ranges):
            segment = grid[sr, c0 : c1 + 1]
            gap_positions = np.flatnonzero(segment != sep_color)
            assert len(gap_positions) <= 1, "each horizontal boundary segment has at most one gap"
            if len(gap_positions) == 1:
                a = index_map[(i, j)]
                b = index_map[(i + 1, j)]
                adj[a].add(b)
                adj[b].add(a)
    for j, sc in enumerate(sep_cols):
        for i, (r0, r1) in enumerate(row_ranges):
            segment = grid[r0 : r1 + 1, sc]
            gap_positions = np.flatnonzero(segment != sep_color)
            assert len(gap_positions) <= 1, "each vertical boundary segment has at most one gap"
            if len(gap_positions) == 1:
                a = index_map[(i, j)]
                b = index_map[(i, j + 1)]
                adj[a].add(b)
                adj[b].add(a)
    return adj


def _path_in_gap_graph(adj: dict[int, set[int]], start: int, end: int) -> list[int]:
    assert _component_is_tree(adj, start, end), "marker component must be a tree"
    queue = deque([start])
    parent = {start: None}
    while queue:
        cur = queue.popleft()
        if cur == end:
            break
        for nxt in sorted(adj[cur]):
            if nxt in parent:
                continue
            parent[nxt] = cur
            queue.append(nxt)
    assert end in parent, "marker regions must be connected by gaps"
    path: list[int] = []
    cur: int | None = end
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the unique region-scale route whose doors are the separator-line gaps.

    Input structure:
    - One-pixel separator bars split the canvas into an m x n array of uniform rectangular regions.
    - Exactly two edge regions share the marker color, and same-colored regions form a visible decoy chain between them.
    - A separator segment may contain one single-pixel gap; only the two regions touching that gap are connected.

    Rule:
    - Treat each separator-delimited region as one coarse cell.
    - Build adjacency only from separator segments that contain a gap, ignoring region colors for movement.
    - Find the unique path between the two marker regions and repaint every region on that path with the marker color, while keeping all separator pixels, gap pixels, and other regions unchanged.

    Key insight:
    - The maze is encoded in the wall breaks, not in the region colors: the long same-color band is a visual lure, while the true route lives on the dual graph of regions linked by gaps.
    """
    g = np.asarray(grid, dtype=int)
    sep_color, sep_rows, sep_cols, row_ranges, col_ranges, regions, index_map = _extract_regions(g)
    marker_color, start, end = _marker_region_indices(regions, (len(row_ranges), len(col_ranges)))
    adj = _gap_graph(g, sep_color, sep_rows, sep_cols, row_ranges, col_ranges, index_map)
    path = _path_in_gap_graph(adj, start, end)

    out = g.copy()
    for idx in path:
        r0, r1, c0, c1 = regions[idx]["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = marker_color
    return out


def _sample_side_markers(rows: int, cols: int) -> tuple[Coord, Coord, str]:
    if random.random() < 0.5:
        start = (random.randrange(1, rows - 1), 0)
        end = (random.randrange(1, rows - 1), cols - 1)
        return start, end, "left-right"
    start = (0, random.randrange(1, cols - 1))
    end = (rows - 1, random.randrange(1, cols - 1))
    return start, end, "top-bottom"


def _make_decoy_path(rows: int, cols: int, start: Coord, end: Coord, mode: str) -> list[Coord]:
    if mode == "left-right":
        spine = random.randrange(1, rows - 1)
        points = [start, (start[0], 1), (spine, 1), (spine, cols - 2), (end[0], cols - 2), end]
    else:
        spine = random.randrange(1, cols - 1)
        points = [start, (1, start[1]), (1, spine), (rows - 2, spine), (rows - 2, end[1]), end]
    path = _stitch_segments(points)
    assert path[0] == start and path[-1] == end
    assert len(path) == len(set(path)), "decoy path must be simple"
    return path


def _make_true_path(rows: int, cols: int, start: Coord, end: Coord, decoy: list[Coord]) -> list[Coord] | None:
    decoy_edges = [_norm_edge(decoy[i], decoy[i + 1]) for i in range(len(decoy) - 1)]
    candidates = decoy_edges[1:-1]
    random.shuffle(candidates)
    for banned_edge in candidates:
        path = _bfs_path(rows, cols, start, end, {banned_edge})
        if path is None:
            continue
        if path == decoy:
            continue
        if set(path) == set(decoy):
            continue
        return path
    return None


def _grow_tree(rows: int, cols: int, path: list[Coord]) -> set[Edge]:
    tree_edges = {_norm_edge(path[i], path[i + 1]) for i in range(len(path) - 1)}
    tree_nodes = set(path)
    target_add = random.randint(*EXTRA_BRANCH_RANGE)
    max_add = rows * cols - len(tree_nodes)
    target_add = min(target_add, max_add)
    while len(tree_nodes) < len(set(path)) + target_add:
        frontier = []
        for node in tree_nodes:
            for nxt in _neighbors4(node, rows, cols):
                if nxt not in tree_nodes:
                    frontier.append((node, nxt))
        if not frontier:
            break
        node, nxt = random.choice(frontier)
        tree_edges.add(_norm_edge(node, nxt))
        tree_nodes.add(nxt)
    return tree_edges


def _sample_cell_sizes(count: int) -> list[int]:
    return [random.randint(*CELL_SIZE_RANGE) for _ in range(count)]


def _ranges_from_sizes(sizes: list[int]) -> tuple[list[tuple[int, int]], list[int]]:
    ranges: list[tuple[int, int]] = []
    bars: list[int] = []
    cursor = 0
    for i, size in enumerate(sizes):
        ranges.append((cursor, cursor + size - 1))
        cursor += size
        if i != len(sizes) - 1:
            bars.append(cursor)
            cursor += 1
    return ranges, bars


def _fix_edge_color_ambiguity(colors: np.ndarray, marker_color: int):
    rows, cols = colors.shape
    edge_cells = []
    for r in range(rows):
        for c in range(cols):
            if r == 0 or c == 0 or r == rows - 1 or c == cols - 1:
                edge_cells.append((r, c))
    counts = Counter(int(colors[r, c]) for r, c in edge_cells)
    other_palette = [c for c in range(10) if c != marker_color]
    for r, c in edge_cells:
        color = int(colors[r, c])
        if color == marker_color:
            continue
        if counts[color] == 2:
            replacement_choices = [cand for cand in other_palette if counts[cand] != 1]
            replacement = random.choice(replacement_choices) if replacement_choices else random.choice(other_palette)
            colors[r, c] = replacement
            counts[color] -= 1
            counts[replacement] += 1


def _render_scene(
    rows: int,
    cols: int,
    row_sizes: list[int],
    col_sizes: list[int],
    region_colors: np.ndarray,
    sep_color: int,
    tree_edges: set[Edge],
) -> np.ndarray:
    row_ranges, sep_rows = _ranges_from_sizes(row_sizes)
    col_ranges, sep_cols = _ranges_from_sizes(col_sizes)
    height = row_ranges[-1][1] + 1
    width = col_ranges[-1][1] + 1
    grid = np.full((height, width), sep_color, dtype=int)
    for r in range(rows):
        r0, r1 = row_ranges[r]
        for c in range(cols):
            c0, c1 = col_ranges[c]
            grid[r0 : r1 + 1, c0 : c1 + 1] = int(region_colors[r, c])

    for a, b in tree_edges:
        if a[0] == b[0]:
            r = a[0]
            c_left = min(a[1], b[1])
            sc = sep_cols[c_left]
            r0, r1 = row_ranges[r]
            gap_r = random.randint(r0, r1)
            gap_color = int(region_colors[a]) if random.random() < 0.5 else int(region_colors[b])
            grid[gap_r, sc] = gap_color
        else:
            r_top = min(a[0], b[0])
            c = a[1]
            sr = sep_rows[r_top]
            c0, c1 = col_ranges[c]
            gap_c = random.randint(c0, c1)
            gap_color = int(region_colors[a]) if random.random() < 0.5 else int(region_colors[b])
            grid[sr, gap_c] = gap_color
    return grid


def _expected_output_from_path(grid: np.ndarray, path_cells: set[Coord], marker_color: int) -> np.ndarray:
    out = grid.copy()
    _sep_color, _sep_rows, _sep_cols, row_ranges, col_ranges, _regions, _index_map = _extract_regions(grid)
    for r, c in path_cells:
        r0, r1 = row_ranges[r]
        c0, c1 = col_ranges[c]
        out[r0 : r1 + 1, c0 : c1 + 1] = marker_color
    return out


def _validate_scene(
    grid: np.ndarray,
    tree_edges: set[Edge],
    true_path: list[Coord],
    decoy_path: list[Coord],
    marker_color: int,
) -> bool:
    try:
        sep_color, sep_rows, sep_cols, row_ranges, col_ranges, regions, index_map = _extract_regions(grid)
        parsed_marker, start_idx, end_idx = _marker_region_indices(regions, (len(row_ranges), len(col_ranges)))
        if parsed_marker != marker_color:
            return False
        adj = _gap_graph(grid, sep_color, sep_rows, sep_cols, row_ranges, col_ranges, index_map)
        parsed_path = _path_in_gap_graph(adj, start_idx, end_idx)
    except Exception:
        return False

    parsed_cells = [regions[idx]["cell"] for idx in parsed_path]
    if parsed_cells != true_path:
        return False

    expected = _expected_output_from_path(grid, set(true_path), marker_color)
    try:
        actual = main(grid)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    decoy_edges = {_norm_edge(decoy_path[i], decoy_path[i + 1]) for i in range(len(decoy_path) - 1)}
    if decoy_edges <= tree_edges:
        return False
    if set(true_path) <= set(decoy_path):
        return False

    edge_marker_cells = [cell for cell in decoy_path if cell[0] in (0, len(row_ranges) - 1) or cell[1] in (0, len(col_ranges) - 1)]
    if len(edge_marker_cells) != 2:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(*ROW_COUNT_RANGE)
    cols = random.randint(*COL_COUNT_RANGE)
    start, end, mode = _sample_side_markers(rows, cols)
    decoy_path = _make_decoy_path(rows, cols, start, end, mode)
    true_path = _make_true_path(rows, cols, start, end, decoy_path)
    if true_path is None:
        return None

    tree_edges = _grow_tree(rows, cols, true_path)
    marker_color, sep_color = random.sample(range(10), 2)
    other_colors = [c for c in range(10) if c not in (marker_color, sep_color)]

    region_colors = np.zeros((rows, cols), dtype=int)
    for r in range(rows):
        for c in range(cols):
            region_colors[r, c] = random.choice(other_colors)
    for cell in decoy_path:
        region_colors[cell] = marker_color
    if any(region_colors[cell] == marker_color for cell in set(true_path) - set(decoy_path)):
        return None

    _fix_edge_color_ambiguity(region_colors, marker_color)
    if any(region_colors[cell] != marker_color for cell in decoy_path):
        return None
    if sum(int(region_colors[r, c] == marker_color) for r in range(rows) for c in range(cols) if r in (0, rows - 1) or c in (0, cols - 1)) != 2:
        return None

    row_sizes = _sample_cell_sizes(rows)
    col_sizes = _sample_cell_sizes(cols)
    grid = _render_scene(rows, cols, row_sizes, col_sizes, region_colors, sep_color, tree_edges)
    if not _validate_scene(grid, tree_edges, true_path, decoy_path, marker_color):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gap-maze puzzle")
