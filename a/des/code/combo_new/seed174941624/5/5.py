import random
from collections import Counter, deque

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
BOX_BG = 0
BOX_NODE = 1
BOX_PATH = 2
GEN_RETRIES = 300


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIR4:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _path_cells_from_waypoints(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for i in range(len(points) - 1):
        r0, c0 = points[i]
        r1, c1 = points[i + 1]
        if r0 != r1 and c0 != c1:
            raise ValueError("polyline must be orthogonal")
        if r0 == r1:
            step = 1 if c1 >= c0 else -1
            for c in range(c0, c1 + step, step):
                if not cells or cells[-1] != (r0, c):
                    cells.append((r0, c))
        else:
            step = 1 if r1 >= r0 else -1
            for r in range(r0, r1 + step, step):
                if not cells or cells[-1] != (r, c0):
                    cells.append((r, c0))
    return cells


def _make_box(shape: tuple[int, int], node_a: tuple[int, int], node_b: tuple[int, int], points: list[tuple[int, int]]) -> np.ndarray:
    h, w = shape
    box = np.zeros((h, w), dtype=int)
    for top, left in (node_a, node_b):
        box[top : top + 2, left : left + 2] = BOX_NODE
    for r, c in _path_cells_from_waypoints(points):
        if box[r, c] != BOX_BG:
            raise ValueError("path overlaps node")
        box[r, c] = BOX_PATH
    return box


TEMPLATE_BOXES = {
    0: _make_box((4, 8), (1, 0), (1, 6), [(1, 2), (1, 5)]),
    1: _make_box((6, 7), (1, 0), (4, 5), [(1, 2), (1, 5), (3, 5)]),
    2: _make_box((6, 8), (1, 0), (4, 6), [(1, 2), (1, 4), (4, 4), (4, 5)]),
    3: _make_box((6, 9), (4, 0), (3, 7), [(4, 2), (4, 5), (1, 5), (1, 7), (2, 7)]),
    4: _make_box((6, 10), (4, 0), (3, 8), [(4, 2), (4, 4), (1, 4), (1, 6), (3, 6), (3, 7)]),
}


def _transform_box(box: np.ndarray, op: int) -> np.ndarray:
    out = np.array(box, copy=True)
    if op >= 4:
        out = np.fliplr(out)
        op -= 4
    out = np.rot90(out, k=op)
    return out


def _edge_line_runs(values: list[int], valid_colors: set[int]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    i = 0
    while i < len(values):
        if values[i] not in valid_colors:
            i += 1
            continue
        start = i
        while i + 1 < len(values) and values[i + 1] in valid_colors:
            i += 1
        runs.append((start, i))
        i += 1
    return runs


def _find_palette(grid: np.ndarray, bg: int) -> tuple[list[int], set[tuple[int, int]]]:
    h, w = grid.shape
    counts = Counter(map(int, grid.flat))
    single_colors = {color for color, count in counts.items() if color != bg and count == 1}
    candidates: list[tuple[list[int], set[tuple[int, int]]]] = []

    for r in (0, h - 1):
        row_vals = [int(v) for v in grid[r]]
        for start, end in _edge_line_runs(row_vals, single_colors):
            colors = row_vals[start : end + 1]
            if 3 <= len(colors) <= 5 and len(set(colors)) == len(colors):
                cells = {(r, c) for c in range(start, end + 1)}
                candidates.append((colors, cells))

    for c in (0, w - 1):
        col_vals = [int(v) for v in grid[:, c]]
        for start, end in _edge_line_runs(col_vals, single_colors):
            colors = col_vals[start : end + 1]
            if 3 <= len(colors) <= 5 and len(set(colors)) == len(colors):
                cells = {(r, c) for r in range(start, end + 1)}
                candidates.append((colors, cells))

    unique_candidates: list[tuple[list[int], set[tuple[int, int]]]] = []
    seen_keys = set()
    for colors, cells in candidates:
        key = tuple(sorted(cells))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique_candidates.append((colors, cells))

    assert len(unique_candidates) == 1, "expected exactly one palette on the grid edge"
    return unique_candidates[0]


def _find_node_color_and_nodes(grid: np.ndarray, bg: int, palette_cells: set[tuple[int, int]]):
    palette_mask = np.zeros(grid.shape, dtype=bool)
    for r, c in palette_cells:
        palette_mask[r, c] = True

    node_candidates = []
    for color in sorted(set(map(int, grid.flat))):
        if color == bg:
            continue
        mask = (grid == color) & ~palette_mask
        comps = _components(mask)
        if not comps:
            continue
        ok = True
        for comp in comps:
            if len(comp) != 4:
                ok = False
                break
            r0, c0, r1, c1 = _bbox(comp)
            if r1 - r0 != 1 or c1 - c0 != 1:
                ok = False
                break
        if ok:
            node_candidates.append((color, comps))

    assert len(node_candidates) == 1, "expected exactly one node color"
    return node_candidates[0]


def _trace_path(comp: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if len(comp) == 1:
        return comp[:]
    pts = set(comp)
    adj: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in comp:
        nbrs = []
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (nr, nc) in pts:
                nbrs.append((nr, nc))
        assert len(nbrs) <= 2, "path cannot branch"
        adj[(r, c)] = nbrs

    ends = [cell for cell, nbrs in adj.items() if len(nbrs) == 1]
    assert len(ends) == 2, "path must have two endpoints"

    ordered = [ends[0]]
    prev = None
    cur = ends[0]
    while True:
        nxts = [nxt for nxt in adj[cur] if nxt != prev]
        if not nxts:
            break
        assert len(nxts) == 1, "path cannot branch while tracing"
        prev, cur = cur, nxts[0]
        ordered.append(cur)
    assert len(ordered) == len(comp), "path trace must visit every pixel"
    return ordered


def _count_turns(comp: list[tuple[int, int]]) -> int:
    ordered = _trace_path(comp)
    if len(ordered) < 2:
        return 0
    directions = []
    for (r0, c0), (r1, c1) in zip(ordered, ordered[1:]):
        directions.append((r1 - r0, c1 - c0))
    return sum(directions[i] != directions[i - 1] for i in range(1, len(directions)))


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be two-dimensional"
    h, w = g.shape
    bg = int(Counter(map(int, g.flat)).most_common(1)[0][0])
    palette_colors, palette_cells = _find_palette(g, bg)
    palette_mask = np.zeros((h, w), dtype=bool)
    for r, c in palette_cells:
        palette_mask[r, c] = True

    node_color, node_comps = _find_node_color_and_nodes(g, bg, palette_cells)
    node_lookup: dict[tuple[int, int], int] = {}
    nodes: dict[int, list[tuple[int, int]]] = {}
    for idx, comp in enumerate(node_comps):
        nodes[idx] = comp
        for cell in comp:
            node_lookup[cell] = idx

    remaining_colors = sorted(set(map(int, g.flat)) - {bg, node_color} - set(palette_colors))
    assert len(remaining_colors) == 1, "expected exactly one path color"
    path_color = remaining_colors[0]
    path_comps = _components((g == path_color) & ~palette_mask)
    assert path_comps, "expected at least one path component"

    out = g.copy()
    for r, c in palette_cells:
        out[r, c] = bg

    node_use_count = Counter()
    for comp in path_comps:
        touching = set()
        for r, c in comp:
            for nr, nc in _neighbors4(r, c, h, w):
                node_id = node_lookup.get((nr, nc))
                if node_id is not None:
                    touching.add(node_id)
        assert len(touching) == 2, "each path must touch exactly two nodes"
        turns = _count_turns(comp)
        assert 0 <= turns < len(palette_colors), "turn count must index the palette"
        new_color = palette_colors[turns]
        for node_id in touching:
            node_use_count[node_id] += 1
            for r, c in nodes[node_id]:
                out[r, c] = new_color
        for r, c in comp:
            out[r, c] = new_color

    assert all(node_use_count[idx] == 1 for idx in range(len(nodes))), "each node must belong to one pair"
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each node pair and its connecting path using the palette color indexed by that path's turn count.

    Input structure:
    - The background is the most common color.
    - Several 2×2 monochrome squares of one shared color act as nodes.
    - A different color forms one-pixel-wide orthogonal paths; each path component touches exactly two nodes and paths do not cross.
    - A 3-to-5-cell palette of distinct single pixels lies on one grid edge in a straight row or column.

    Rule:
    - Locate the edge palette and read it in geometric order: left-to-right for horizontal, top-to-bottom for vertical.
    - For each path component, trace it from one endpoint to the other and count how many times its direction switches between horizontal and vertical.
    - Use that turn count as a zero-based index into the palette, then recolor both endpoint nodes and every pixel of the path to the indexed palette color.
    - Erase the palette itself back to the background color.

    Key insight:
    - The decisive attribute is not a path's length but its polyline geometry: the number of right-angle turns selects a visible lookup color on the edge palette.
    """
    return _solve(grid)


def _choose_colors(palette_len: int) -> tuple[int, int, int, list[int]]:
    colors = list(range(10))
    bg = random.choice(colors)
    colors.remove(bg)
    node_color = random.choice(colors)
    colors.remove(node_color)
    path_color = random.choice(colors)
    colors.remove(path_color)
    palette = random.sample(colors, palette_len)
    return bg, node_color, path_color, palette


def _pack_boxes(boxes: list[np.ndarray]) -> tuple[list[tuple[int, int]], int, int] | None:
    max_w = max(box.shape[1] for box in boxes)
    candidate_widths = list(range(max(max_w, 16), 27))
    random.shuffle(candidate_widths)
    for inner_w in candidate_widths:
        positions: list[tuple[int, int]] = []
        x = 0
        y = 0
        row_h = 0
        ok = True
        for box in boxes:
            h, w = box.shape
            if x > 0 and x + w > inner_w:
                y += row_h + 2
                x = 0
                row_h = 0
            if w > inner_w:
                ok = False
                break
            positions.append((y, x))
            x += w + 2
            row_h = max(row_h, h)
        if not ok:
            continue
        inner_h = y + row_h
        if inner_h <= 26:
            used_w = 0
            for (top, left), box in zip(positions, boxes):
                used_w = max(used_w, left + box.shape[1])
            return positions, inner_h, used_w
    return None


def _place_palette(grid: np.ndarray, bg: int, palette: list[int]) -> bool:
    h, w = grid.shape
    side = random.choice(["top", "bottom", "left", "right"])
    length = len(palette)

    if side in ("top", "bottom"):
        row = 0 if side == "top" else h - 1
        starts = list(range(0, w - length + 1))
        random.shuffle(starts)
        for start in starts:
            if np.all(grid[row, start : start + length] == bg):
                grid[row, start : start + length] = palette
                return True
    else:
        col = 0 if side == "left" else w - 1
        starts = list(range(0, h - length + 1))
        random.shuffle(starts)
        for start in starts:
            if np.all(grid[start : start + length, col] == bg):
                grid[start : start + length, col] = np.array(palette, dtype=int)
                return True
    return False


def _gen_attempt() -> np.ndarray | None:
    palette_len = random.randint(3, 5)
    pair_count = random.randint(max(2, palette_len - 1), palette_len)
    turn_counts = random.sample(list(range(palette_len)), pair_count)
    boxes = [_transform_box(TEMPLATE_BOXES[turns], random.randint(0, 7)) for turns in turn_counts]
    random.shuffle(boxes)

    packed = _pack_boxes(boxes)
    if packed is None:
        return None
    positions, inner_h, inner_w = packed
    extra_h = random.randint(0, 30 - (inner_h + 2))
    extra_w = random.randint(0, 30 - (inner_w + 2))
    h = inner_h + 2 + extra_h
    w = inner_w + 2 + extra_w
    if h > 30 or w > 30:
        return None

    bg, node_color, path_color, palette = _choose_colors(palette_len)
    grid = np.full((h, w), bg, dtype=int)
    offset_r = 1 + random.randint(0, h - 2 - inner_h)
    offset_c = 1 + random.randint(0, w - 2 - inner_w)

    for (top, left), box in zip(positions, boxes):
        r0 = offset_r + top
        c0 = offset_c + left
        patch = grid[r0 : r0 + box.shape[0], c0 : c0 + box.shape[1]]
        if np.any((box != BOX_BG) & (patch != bg)):
            return None
        patch[box == BOX_NODE] = node_color
        patch[box == BOX_PATH] = path_color

    if not _place_palette(grid, bg, palette):
        return None

    if Counter(map(int, grid.flat)).most_common(1)[0][0] != bg:
        return None

    try:
        _solve(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
