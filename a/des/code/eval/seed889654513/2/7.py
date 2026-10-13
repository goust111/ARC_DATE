from __future__ import annotations

import numpy as np


DIRS = {
    "N": (-1, 0),
    "S": (1, 0),
    "W": (0, -1),
    "E": (0, 1),
}
OPPOSITE = {"N": "S", "S": "N", "W": "E", "E": "W"}
COMPRESS_DELTA = {"N": (-2, 0), "S": (2, 0), "W": (0, -2), "E": (0, 2)}


def _most_common_color(arr: np.ndarray) -> int:
    vals, cnts = np.unique(np.asarray(arr, dtype=int), return_counts=True)
    return int(vals[int(cnts.argmax())])


def _bg_and_grey(grid: np.ndarray) -> tuple[int, int]:
    bg = _most_common_color(grid)
    fg = grid[grid != bg]
    assert fg.size > 0, "grid must contain a foreground graph"
    grey = _most_common_color(fg)
    return int(bg), int(grey)


def _follow_to_node(
    grid: np.ndarray,
    start: tuple[int, int],
    dr: int,
    dc: int,
    grey: int,
    pos_to_idx: dict[tuple[int, int], int],
) -> tuple[int, int] | None:
    h, w = grid.shape
    r, c = start[0] + dr, start[1] + dc
    length = 0
    while 0 <= r < h and 0 <= c < w and int(grid[r, c]) == grey:
        r += dr
        c += dc
        length += 1
    if length == 0 or not (0 <= r < h and 0 <= c < w):
        return None
    if (int(r), int(c)) not in pos_to_idx:
        return None
    return pos_to_idx[(int(r), int(c))], int(length)


def _extract_graph(
    grid: np.ndarray,
) -> tuple[int, int, list[tuple[int, int, int]], list[list[tuple[int, str]]]]:
    g = np.asarray(grid, dtype=int)
    bg, grey = _bg_and_grey(g)

    nodes: list[tuple[int, int, int]] = []
    for r, c in np.argwhere((g != bg) & (g != grey)):
        nodes.append((int(r), int(c), int(g[r, c])))
    assert nodes, "expected at least one colored node"

    pos_to_idx = {(r, c): i for i, (r, c, _color) in enumerate(nodes)}
    adjacency: list[list[tuple[int, str]]] = [[] for _ in nodes]
    seen_edges: set[tuple[int, int]] = set()

    for i, (r, c, _color) in enumerate(nodes):
        for name, (dr, dc) in DIRS.items():
            found = _follow_to_node(g, (r, c), dr, dc, grey, pos_to_idx)
            if found is None:
                continue
            j, _length = found
            if i == j:
                continue
            key = (i, j) if i < j else (j, i)
            if key in seen_edges:
                continue
            seen_edges.add(key)
            adjacency[i].append((j, name))
            adjacency[j].append((i, OPPOSITE[name]))

    return int(bg), int(grey), nodes, adjacency


def _connected_components(adjacency: list[list[tuple[int, str]]]) -> list[list[int]]:
    seen = set()
    comps: list[list[int]] = []
    for start in range(len(adjacency)):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        comp: list[int] = []
        while stack:
            cur = stack.pop()
            comp.append(int(cur))
            for nxt, _name in adjacency[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(int(nxt))
        comps.append(sorted(comp))
    return comps


def _assign_component_positions(
    comp: list[int],
    nodes: list[tuple[int, int, int]],
    adjacency: list[list[tuple[int, str]]],
) -> dict[int, tuple[int, int]]:
    root = min(comp, key=lambda idx: (nodes[idx][0], nodes[idx][1]))
    placed = {int(root): (0, 0)}
    stack = [int(root)]

    while stack:
        cur = stack.pop()
        base_r, base_c = placed[cur]
        for nxt, direction in adjacency[cur]:
            dr, dc = COMPRESS_DELTA[direction]
            cand = (int(base_r + dr), int(base_c + dc))
            if nxt in placed:
                assert placed[nxt] == cand, "orthogonal embedding must be consistent"
            else:
                placed[int(nxt)] = cand
                stack.append(int(nxt))

    assert set(placed) == set(comp), "component must be fully reachable"
    coords = list(placed.values())
    assert len(coords) == len(set(coords)), "compressed embedding must not collapse distinct nodes"
    return placed


def _render_component_tile(
    comp: list[int],
    placed: dict[int, tuple[int, int]],
    nodes: list[tuple[int, int, int]],
    adjacency: list[list[tuple[int, str]]],
    bg: int,
    grey: int,
) -> np.ndarray:
    rs = [placed[idx][0] for idx in comp]
    cs = [placed[idx][1] for idx in comp]
    rmin, rmax = min(rs), max(rs)
    cmin, cmax = min(cs), max(cs)

    tile = np.full((rmax - rmin + 1, cmax - cmin + 1), bg, dtype=int)
    drawn_edges: set[tuple[int, int]] = set()

    for idx in comp:
        rr, cc = placed[idx]
        tile[rr - rmin, cc - cmin] = int(nodes[idx][2])

    for idx in comp:
        for nxt, _direction in adjacency[idx]:
            key = (idx, nxt) if idx < nxt else (nxt, idx)
            if key in drawn_edges:
                continue
            drawn_edges.add(key)
            r1, c1 = placed[idx]
            r2, c2 = placed[nxt]
            assert abs(r1 - r2) + abs(c1 - c2) == 2, "compressed edge must span one grey pixel"
            tile[(r1 + r2) // 2 - rmin, (c1 + c2) // 2 - cmin] = grey

    return tile


def _pack_component_tiles(
    records: list[dict[str, object]],
    bg: int,
) -> np.ndarray:
    assert records, "expected at least one component"
    records = sorted(records, key=lambda rec: (rec["top"], rec["left"]))  # type: ignore[index]

    row_groups: list[list[dict[str, object]]] = []
    current: list[dict[str, object]] = []
    current_bottom = None
    for rec in records:
        top = int(rec["top"])  # type: ignore[index]
        bottom = int(rec["bottom"])  # type: ignore[index]
        if not current or top <= int(current_bottom):
            current.append(rec)
            current_bottom = bottom if current_bottom is None else max(int(current_bottom), bottom)
        else:
            row_groups.append(current)
            current = [rec]
            current_bottom = bottom
    if current:
        row_groups.append(current)

    row_tiles: list[np.ndarray] = []
    for group in row_groups:
        group.sort(key=lambda rec: (rec["left"], rec["top"]))  # type: ignore[index]
        base_top = min(int(rec["top"]) for rec in group)  # type: ignore[index]
        row_h = max(
            int(rec["top"]) - base_top + int(rec["tile"].shape[0])  # type: ignore[index]
            for rec in group
        )
        row_w = sum(int(rec["tile"].shape[1]) for rec in group) + max(0, len(group) - 1)  # type: ignore[index]
        row_tile = np.full((row_h, row_w), bg, dtype=int)

        x = 0
        for rec in group:
            tile = rec["tile"]  # type: ignore[index]
            y = int(rec["top"]) - base_top  # type: ignore[index]
            h, w = tile.shape
            mask = tile != bg
            row_tile[y : y + h, x : x + w][mask] = tile[mask]
            x += int(w) + 1
        row_tiles.append(row_tile)

    full_h = sum(tile.shape[0] for tile in row_tiles) + max(0, len(row_tiles) - 1) + 2
    full_w = max(tile.shape[1] for tile in row_tiles) + 2
    out = np.full((full_h, full_w), bg, dtype=int)
    y = 1
    for row_tile in row_tiles:
        h, w = row_tile.shape
        out[y : y + h, 1 : 1 + w] = row_tile
        y += h + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress sparse orthogonal graphs made of colored nodes and grey path segments.

    Each maximal grey segment between adjacent structural nodes is reduced to a
    single grey pixel while node colors, connection directions, and turn
    structure are preserved. Disconnected components are compacted separately
    and then packed into a minimal output canvas with a one-cell background
    border.
    """
    g = np.asarray(grid, dtype=int)
    bg, grey, nodes, adjacency = _extract_graph(g)
    comps = _connected_components(adjacency)

    records: list[dict[str, object]] = []
    for comp in comps:
        placed = _assign_component_positions(comp, nodes, adjacency)
        tile = _render_component_tile(comp, placed, nodes, adjacency, bg, grey)
        top = min(nodes[idx][0] for idx in comp)
        left = min(nodes[idx][1] for idx in comp)
        bottom = max(nodes[idx][0] for idx in comp)
        records.append(
            {
                "tile": tile,
                "top": int(top),
                "left": int(left),
                "bottom": int(bottom),
            }
        )

    return _pack_component_tiles(records, bg)


TEMPLATES = [
    (
        [(0, 0), (0, 1)],
        [((0, 0), (0, 1))],
    ),
    (
        [(0, 0), (0, 1), (1, 1)],
        [((0, 0), (0, 1)), ((0, 1), (1, 1))],
    ),
    (
        [(0, 0), (0, 1), (1, 1), (1, 2)],
        [((0, 0), (0, 1)), ((0, 1), (1, 1)), ((1, 1), (1, 2))],
    ),
    (
        [(0, 1), (1, 1), (1, 0), (1, 2)],
        [((0, 1), (1, 1)), ((1, 1), (1, 0)), ((1, 1), (1, 2))],
    ),
    (
        [(1, 1), (0, 1), (2, 1), (1, 0), (1, 2)],
        [((1, 1), (0, 1)), ((1, 1), (2, 1)), ((1, 1), (1, 0)), ((1, 1), (1, 2))],
    ),
    (
        [(0, 0), (0, 1), (1, 1), (1, 0)],
        [((0, 0), (0, 1)), ((0, 1), (1, 1)), ((1, 1), (1, 0)), ((1, 0), (0, 0))],
    ),
    (
        [(0, 0), (0, 1), (1, 1), (1, 2), (2, 1)],
        [((0, 0), (0, 1)), ((0, 1), (1, 1)), ((1, 1), (1, 2)), ((1, 1), (2, 1))],
    ),
]


def _transform_component(
    nodes: list[tuple[int, int]],
    edges: list[tuple[tuple[int, int], tuple[int, int]]],
    rng: np.random.Generator,
) -> tuple[list[tuple[int, int]], list[tuple[tuple[int, int], tuple[int, int]]]]:
    rot = int(rng.integers(0, 4))
    flip = bool(rng.integers(0, 2))

    def tr(pt: tuple[int, int]) -> tuple[int, int]:
        r, c = pt
        for _ in range(rot):
            r, c = c, -r
        if flip:
            c = -c
        return int(r), int(c)

    new_nodes = [tr(pt) for pt in nodes]
    new_edges = [(tr(a), tr(b)) for a, b in edges]
    rmin = min(r for r, _ in new_nodes)
    cmin = min(c for _, c in new_nodes)
    shift = lambda pt: (int(pt[0] - rmin), int(pt[1] - cmin))
    new_nodes = [shift(pt) for pt in new_nodes]
    new_edges = [(shift(a), shift(b)) for a, b in new_edges]
    return new_nodes, new_edges


def _sample_node_colors(
    count: int,
    palette: list[int],
    rng: np.random.Generator,
) -> list[int]:
    k = int(rng.integers(2, min(len(palette), 4) + 1))
    active = [int(v) for v in rng.choice(np.array(palette, dtype=int), size=k, replace=False)]
    colors = [int(v) for v in rng.choice(np.array(active, dtype=int), size=count, replace=True)]
    if count >= 2 and len(set(colors)) == 1:
        colors[0] = active[0]
        colors[1] = active[1 % len(active)]
    return colors


def _render_expanded_component(
    nodes: list[tuple[int, int]],
    edges: list[tuple[tuple[int, int], tuple[int, int]]],
    bg: int,
    grey: int,
    palette: list[int],
    rng: np.random.Generator,
) -> np.ndarray:
    rows = sorted({r for r, _ in nodes})
    cols = sorted({c for _, c in nodes})

    row_map = {rows[0]: 0}
    for i in range(1, len(rows)):
        row_map[rows[i]] = row_map[rows[i - 1]] + int(rng.integers(3, 8))
    col_map = {cols[0]: 0}
    for i in range(1, len(cols)):
        col_map[cols[i]] = col_map[cols[i - 1]] + int(rng.integers(3, 8))

    height = row_map[rows[-1]] + 1
    width = col_map[cols[-1]] + 1
    tile = np.full((height, width), bg, dtype=int)

    node_colors = _sample_node_colors(len(nodes), palette, rng)
    for (r, c), color in zip(nodes, node_colors, strict=False):
        tile[row_map[r], col_map[c]] = int(color)

    for (a, b) in edges:
        r1, c1 = row_map[a[0]], col_map[a[1]]
        r2, c2 = row_map[b[0]], col_map[b[1]]
        if r1 == r2:
            lo, hi = sorted((c1, c2))
            tile[r1, lo + 1 : hi] = grey
        else:
            lo, hi = sorted((r1, r2))
            tile[lo + 1 : hi, c1] = grey

    return tile


def _compose_input(
    tiles: list[np.ndarray],
    bg: int,
    rng: np.random.Generator,
) -> np.ndarray:
    margin_top = int(rng.integers(1, 5))
    margin_bottom = int(rng.integers(1, 5))
    margin_left = int(rng.integers(1, 5))
    margin_right = int(rng.integers(1, 5))

    if len(tiles) == 1:
        tile = tiles[0]
        out = np.full(
            (margin_top + tile.shape[0] + margin_bottom, margin_left + tile.shape[1] + margin_right),
            bg,
            dtype=int,
        )
        out[margin_top : margin_top + tile.shape[0], margin_left : margin_left + tile.shape[1]] = tile
        return out

    order = list(range(len(tiles)))
    rng.shuffle(order)
    tiles = [tiles[idx] for idx in order]
    same_row = bool(rng.integers(0, 2))
    if same_row:
        gap = int(rng.integers(2, 7))
        band_h = max(tile.shape[0] for tile in tiles)
        width = margin_left + sum(tile.shape[1] for tile in tiles) + gap * (len(tiles) - 1) + margin_right
        height = margin_top + band_h + margin_bottom
        out = np.full((height, width), bg, dtype=int)
        x = margin_left
        for tile in tiles:
            out[margin_top : margin_top + tile.shape[0], x : x + tile.shape[1]] = tile
            x += tile.shape[1] + gap
        return out

    gap = int(rng.integers(2, 7))
    band_w = max(tile.shape[1] for tile in tiles)
    width = margin_left + band_w + margin_right
    height = margin_top + sum(tile.shape[0] for tile in tiles) + gap * (len(tiles) - 1) + margin_bottom
    out = np.full((height, width), bg, dtype=int)
    y = margin_top
    for tile in tiles:
        x = margin_left + int(rng.integers(0, band_w - tile.shape[1] + 1))
        out[y : y + tile.shape[0], x : x + tile.shape[1]] = tile
        y += tile.shape[0] + gap
    return out


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    bg_choices = [0, 1, 8]
    grey_choices = [5, 6]
    bg = int(rng.choice(np.array(bg_choices, dtype=int)))
    grey = int(rng.choice(np.array([v for v in grey_choices if v != bg], dtype=int)))
    palette = [v for v in range(10) if v not in (bg, grey)]

    comp_count = 1 if rng.random() < 0.65 else 2
    tiles: list[np.ndarray] = []

    for _ in range(comp_count):
        base_nodes, base_edges = TEMPLATES[int(rng.integers(0, len(TEMPLATES)))]
        nodes, edges = _transform_component(base_nodes, base_edges, rng)
        tile = _render_expanded_component(nodes, edges, bg, grey, palette, rng)
        tiles.append(tile)

    grid = _compose_input(tiles, bg, rng)

    try:
        found_bg, found_grey, nodes, adjacency = _extract_graph(grid)
        if found_bg != bg or found_grey != grey:
            return None
        if len(nodes) < 2 or not any(adjacency):
            return None
        out = main(grid)
        if not np.array_equal(main(out), out):
            return None
    except AssertionError:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(300):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after 300 attempts")
