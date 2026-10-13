from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


CELL_GRID_MIN = 3
CELL_GRID_MAX = 5
CELL_SIZE_MIN = 3
CELL_SIZE_MAX = 5
MAX_GEN_ATTEMPTS = 500
PALETTE = tuple(range(10))


Cell = tuple[int, int]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, grid.ravel()))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def _segments(length: int, dividers: list[int]) -> list[tuple[int, int]]:
    parts: list[tuple[int, int]] = []
    start = 0
    for divider in dividers:
        assert start < divider
        parts.append((start, divider))
        start = divider + 1
    assert start < length
    parts.append((start, length))
    return parts


def _divider_grid(g: np.ndarray, bg: int) -> tuple[int, list[int], list[int]]:
    h, w = g.shape
    candidates: list[tuple[int, list[int], list[int]]] = []
    for color in _colors_present(g, exclude={bg}):
        row_hits = [r for r in range(h) if int(np.count_nonzero(g[r, :] == color)) > w // 2]
        col_hits = [c for c in range(w) if int(np.count_nonzero(g[:, c] == color)) > h // 2]
        if row_hits and col_hits:
            candidates.append((int(color), row_hits, col_hits))

    assert candidates, "expected a divider color"
    candidates.sort(key=lambda item: (len(item[1]) + len(item[2]), len(item[1]), len(item[2])), reverse=True)
    best = candidates[0]
    if len(candidates) > 1:
        best_score = len(best[1]) + len(best[2])
        second_score = len(candidates[1][1]) + len(candidates[1][2])
        assert best_score > second_score, "divider color must be unique"
    return best


def _neighbors(cell: Cell, rows: int, cols: int) -> list[Cell]:
    r, c = cell
    out: list[Cell] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < rows:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < cols:
        out.append((r, c + 1))
    return out


def _component(start: Cell, adjacency: dict[Cell, list[Cell]]) -> list[Cell]:
    queue = deque([start])
    seen = {start}
    order: list[Cell] = []
    while queue:
        cur = queue.popleft()
        order.append(cur)
        for nxt in adjacency[cur]:
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return order


def _path_directions(path: list[Cell]) -> list[str]:
    dirs: list[str] = []
    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        if r1 == r0 - 1:
            dirs.append("up")
        elif r1 == r0 + 1:
            dirs.append("down")
        elif c1 == c0 - 1:
            dirs.append("left")
        elif c1 == c0 + 1:
            dirs.append("right")
        else:
            raise AssertionError("non-adjacent path step")
    return dirs


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    bg = _get_bg(g)
    divider_color, divider_rows, divider_cols = _divider_grid(g, bg)
    row_segments = _segments(h, divider_rows)
    col_segments = _segments(w, divider_cols)

    row_sizes = {r1 - r0 for r0, r1 in row_segments}
    col_sizes = {c1 - c0 for c0, c1 in col_segments}
    assert len(row_sizes) == 1 and len(col_sizes) == 1

    cell_rows = len(row_segments)
    cell_cols = len(col_segments)
    seeds: dict[Cell, tuple[int, int, int]] = {}
    adjacency: dict[Cell, list[Cell]] = {(r, c): [] for r in range(cell_rows) for c in range(cell_cols)}

    for rr, (r0, r1) in enumerate(row_segments):
        for cc, (c0, c1) in enumerate(col_segments):
            cell = g[r0:r1, c0:c1]
            mask = (cell != bg) & (cell != divider_color)
            coords = np.argwhere(mask)
            assert len(coords) <= 1, "each cell may contain at most one seed pixel"
            if len(coords) == 1:
                sr, sc = map(int, coords[0])
                seeds[(rr, cc)] = (r0 + sr, c0 + sc, int(cell[sr, sc]))

    for rr, (r0, r1) in enumerate(row_segments):
        for cc in range(cell_cols - 1):
            wall_col = divider_cols[cc]
            wall = g[r0:r1, wall_col]
            holes = np.flatnonzero(wall != divider_color)
            assert len(holes) <= 1, "each wall segment may have at most one visible gap"
            if len(holes) == 1:
                assert int(wall[holes[0]]) == bg, "gap pixels must use background color"
                a = (rr, cc)
                b = (rr, cc + 1)
                adjacency[a].append(b)
                adjacency[b].append(a)

    for rr in range(cell_rows - 1):
        wall_row = divider_rows[rr]
        for cc, (c0, c1) in enumerate(col_segments):
            wall = g[wall_row, c0:c1]
            holes = np.flatnonzero(wall != divider_color)
            assert len(holes) <= 1, "each wall segment may have at most one visible gap"
            if len(holes) == 1:
                assert int(wall[holes[0]]) == bg, "gap pixels must use background color"
                a = (rr, cc)
                b = (rr + 1, cc)
                adjacency[a].append(b)
                adjacency[b].append(a)

    for cell, neighbors in adjacency.items():
        uniq = sorted(set(neighbors))
        assert len(uniq) == len(neighbors), f"duplicate wall edge at {cell}"
        adjacency[cell] = uniq
        assert len(uniq) <= 2, "routed cells must form simple chains"

    return {
        "grid": g,
        "bg": bg,
        "divider_color": divider_color,
        "divider_rows": divider_rows,
        "divider_cols": divider_cols,
        "row_segments": row_segments,
        "col_segments": col_segments,
        "cell_rows": cell_rows,
        "cell_cols": cell_cols,
        "seeds": seeds,
        "adjacency": adjacency,
    }


def _routes(info: dict) -> list[dict]:
    adjacency: dict[Cell, list[Cell]] = info["adjacency"]
    seeds: dict[Cell, tuple[int, int, int]] = info["seeds"]
    seen: set[Cell] = set()
    routes: list[dict] = []

    for cell in adjacency:
        if cell in seen:
            continue
        comp = _component(cell, adjacency)
        seen.update(comp)
        seed_cells = [node for node in comp if node in seeds]
        edge_count = sum(len(adjacency[node]) for node in comp) // 2

        if edge_count == 0:
            if seed_cells:
                assert len(seed_cells) == 1
                routes.append({"seed_cell": seed_cells[0], "path": [seed_cells[0]], "sink_cell": seed_cells[0]})
            continue

        assert len(seed_cells) == 1, "each visible chain must carry exactly one seed"
        endpoints = [node for node in comp if len(adjacency[node]) == 1]
        assert len(endpoints) == 2, "visible chains must be simple paths"
        seed = seed_cells[0]
        assert seed in endpoints, "the seed must sit at one endpoint of its chain"

        path = [seed]
        prev: Cell | None = None
        cur = seed
        while True:
            nxts = [node for node in adjacency[cur] if node != prev]
            if not nxts:
                break
            assert len(nxts) == 1
            prev, cur = cur, nxts[0]
            path.append(cur)

        routes.append({"seed_cell": seed, "path": path, "sink_cell": path[-1]})

    sinks = [route["sink_cell"] for route in routes]
    assert len(sinks) == len(set(sinks)), "distinct seeds must end in distinct sink cells"
    return routes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Let each colored pixel traverse the chain of wall gaps and then flood its terminal gapless cell.

    Input structure:
    - Full horizontal and vertical divider bars split the canvas into equal rectangular cells.
    - Some wall segments contain a single missing pixel, and the visible gaps form disjoint cell-to-cell chains.
    - A few cell interiors contain exactly one colored pixel; the other cell interiors are empty.
    - Each gap chain carries at most one seed, and the chain topology uniquely determines that seed's terminal cell.

    Rule:
    - Read each seed pixel as moving cell by cell along the unique chain attached to its cell.
    - Repeatedly pass the seed through the next wall gap into the neighboring cell until the chain ends; a seed in an isolated cell stops immediately.
    - Clear every seed from its original location, then fill the terminal cell's whole interior with that seed's color.
    - Preserve the divider bars and their gap pixels exactly; any cell that receives no seed stays background.

    Key insight:
    - The missing wall pixels are not noise: they define relay chains across the divider grid, so the answer depends on each seed's final destination cell rather than on where the seed started.
    """
    info = _analyze_grid(grid)
    g = info["grid"]
    bg = info["bg"]
    row_segments = info["row_segments"]
    col_segments = info["col_segments"]
    seeds = info["seeds"]

    out = g.copy()
    for sr, sc, _color in seeds.values():
        out[sr, sc] = bg

    for route in _routes(info):
        seed_cell = route["seed_cell"]
        sink_r, sink_c = route["sink_cell"]
        _sr, _sc, color = seeds[seed_cell]
        r0, r1 = row_segments[sink_r]
        c0, c1 = col_segments[sink_c]
        out[r0:r1, c0:c1] = color

    return out


def _grid_geometry() -> tuple[int, int, int, int]:
    for _ in range(100):
        cell_rows = random.randint(CELL_GRID_MIN, CELL_GRID_MAX)
        cell_cols = random.randint(CELL_GRID_MIN, CELL_GRID_MAX)
        cell_h = random.randint(CELL_SIZE_MIN, CELL_SIZE_MAX)
        cell_w = random.randint(CELL_SIZE_MIN, CELL_SIZE_MAX)
        height = cell_rows * cell_h + (cell_rows - 1)
        width = cell_cols * cell_w + (cell_cols - 1)
        if height <= 30 and width <= 30:
            return cell_rows, cell_cols, cell_h, cell_w
    raise RuntimeError("failed to sample valid grid geometry")


def _segment_layout(cell_count: int, cell_size: int) -> tuple[list[tuple[int, int]], list[int]]:
    segments: list[tuple[int, int]] = []
    dividers: list[int] = []
    for idx in range(cell_count):
        start = idx * (cell_size + 1)
        end = start + cell_size
        segments.append((start, end))
        if idx + 1 < cell_count:
            dividers.append(end)
    return segments, dividers


def _blank_grid(cell_rows: int, cell_cols: int, cell_h: int, cell_w: int, bg: int, divider: int) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]], list[int], list[int]]:
    row_segments, divider_rows = _segment_layout(cell_rows, cell_h)
    col_segments, divider_cols = _segment_layout(cell_cols, cell_w)
    height = row_segments[-1][1]
    width = col_segments[-1][1]
    grid = np.full((height, width), bg, dtype=int)
    for r in divider_rows:
        grid[r, :] = divider
    for c in divider_cols:
        grid[:, c] = divider
    return grid, row_segments, col_segments, divider_rows, divider_cols


def _random_walk_path(start: Cell, length: int, available: set[Cell], cell_rows: int, cell_cols: int) -> list[Cell] | None:
    path = [start]
    used = {start}
    prev_dir: str | None = None

    for step in range(length):
        cur = path[-1]
        candidates = [nb for nb in _neighbors(cur, cell_rows, cell_cols) if nb in available and nb not in used]
        if not candidates:
            return None

        if prev_dir is not None and step + 1 < length:
            bends = [nb for nb in candidates if _path_directions([path[-2], cur, nb])[-1] != prev_dir]
            if bends and random.random() < 0.65:
                nxt = random.choice(bends)
            else:
                nxt = random.choice(candidates)
        else:
            nxt = random.choice(candidates)

        if len(path) >= 2:
            prev_dir = _path_directions(path[-2:] + [nxt])[-1]
        else:
            prev_dir = _path_directions([cur, nxt])[0]
        path.append(nxt)
        used.add(nxt)

    return path


def _sample_paths(cell_rows: int, cell_cols: int) -> tuple[list[list[Cell]], bool]:
    cells = {(r, c) for r in range(cell_rows) for c in range(cell_cols)}
    max_paths = min(4, max(2, (cell_rows * cell_cols) // 3))
    path_count = random.randint(2, max_paths)
    include_isolated = random.random() < 0.35
    need_bend = random.random() < 0.6

    available = set(cells)
    paths: list[list[Cell]] = []
    has_long = False

    for idx in range(path_count):
        remaining = path_count - idx - 1
        free = len(available)
        if free <= remaining:
            raise RuntimeError("insufficient cells for remaining paths")

        if include_isolated and idx == path_count - 1:
            length = 0
        else:
            min_len = 1
            if not has_long:
                min_len = 2
            max_len = min(4, free - 1 - remaining)
            if max_len < min_len:
                raise RuntimeError("cannot reserve a nontrivial path")
            length = random.randint(min_len, max_len)

        path: list[Cell] | None = None
        for _ in range(80):
            start = random.choice(tuple(available))
            if length == 0:
                path = [start]
            else:
                attempt = _random_walk_path(start, length, available, cell_rows, cell_cols)
                if attempt is not None:
                    path = attempt
            if path is not None:
                break
        if path is None:
            raise RuntimeError("failed to place path")

        for cell in path:
            available.remove(cell)
        paths.append(path)
        has_long |= len(path) >= 3

    directions = {d for path in paths for d in _path_directions(path)}
    has_bend = any(len(set(_path_directions(path))) > 1 or any(a != b for a, b in zip(_path_directions(path), _path_directions(path)[1:])) for path in paths)
    if not has_long:
        raise RuntimeError("missing a true cascade path")
    if len(directions) < 2:
        raise RuntimeError("missing directional variety")
    if need_bend and not has_bend:
        raise RuntimeError("missing bend coverage")

    return paths, include_isolated


def _paint_gap(grid: np.ndarray, a: Cell, b: Cell, row_segments: list[tuple[int, int]], col_segments: list[tuple[int, int]], divider_rows: list[int], divider_cols: list[int], bg: int) -> None:
    ar, ac = a
    br, bc = b
    if ar == br:
        wall_col = divider_cols[min(ac, bc)]
        r0, r1 = row_segments[ar]
        grid[random.randint(r0, r1 - 1), wall_col] = bg
        return

    assert ac == bc
    wall_row = divider_rows[min(ar, br)]
    c0, c1 = col_segments[ac]
    grid[wall_row, random.randint(c0, c1 - 1)] = bg


def _paint_seed(grid: np.ndarray, cell: Cell, color: int, row_segments: list[tuple[int, int]], col_segments: list[tuple[int, int]]) -> tuple[int, int]:
    rr, cc = cell
    r0, r1 = row_segments[rr]
    c0, c1 = col_segments[cc]
    sr = random.randint(r0, r1 - 1)
    sc = random.randint(c0, c1 - 1)
    grid[sr, sc] = color
    return sr, sc


def _planned_output(inp: np.ndarray, paths: list[list[Cell]], colors: list[int], row_segments: list[tuple[int, int]], col_segments: list[tuple[int, int]]) -> np.ndarray:
    out = inp.copy()
    bg = _get_bg(inp)
    for path in paths:
        seed = path[0]
        r0, r1 = row_segments[seed[0]]
        c0, c1 = col_segments[seed[1]]
        coords = np.argwhere((out[r0:r1, c0:c1] != bg))
        assert len(coords) == 1
        sr, sc = map(int, coords[0])
        out[r0 + sr, c0 + sc] = bg

    for path, color in zip(paths, colors):
        sink = path[-1]
        r0, r1 = row_segments[sink[0]]
        c0, c1 = col_segments[sink[1]]
        out[r0:r1, c0:c1] = int(color)
    return out


def _oracle(inp: np.ndarray, paths: list[list[Cell]], colors: list[int], include_isolated: bool) -> bool:
    try:
        info = _analyze_grid(inp)
        routes = _routes(info)
        parsed = {(route["seed_cell"], route["sink_cell"], tuple(route["path"])) for route in routes}
        planned = {(path[0], path[-1], tuple(path)) for path in paths}
        if parsed != planned:
            return False

        if include_isolated:
            if not any(len(path) == 1 for path in paths):
                return False
        if not any(len(path) >= 3 for path in paths):
            return False

        all_dirs = {d for path in paths for d in _path_directions(path)}
        if len(all_dirs) < 2:
            return False

        expected = _planned_output(inp, paths, colors, info["row_segments"], info["col_segments"])
        actual = main(inp)
        if not np.array_equal(expected, actual):
            return False
        if np.array_equal(inp, actual):
            return False

        bg = info["bg"]
        for route in routes:
            seed_cell = route["seed_cell"]
            sink_cell = route["sink_cell"]
            _sr, _sc, color = info["seeds"][seed_cell]
            r0, r1 = info["row_segments"][sink_cell[0]]
            c0, c1 = info["col_segments"][sink_cell[1]]
            if not np.all(actual[r0:r1, c0:c1] == color):
                return False
            if len(route["path"]) > 1:
                sr0, sr1 = info["row_segments"][seed_cell[0]]
                sc0, sc1 = info["col_segments"][seed_cell[1]]
                if np.any(actual[sr0:sr1, sc0:sc1] != bg):
                    return False
        return True
    except AssertionError:
        return False


def _gen_attempt() -> np.ndarray | None:
    cell_rows, cell_cols, cell_h, cell_w = _grid_geometry()
    try:
        paths, include_isolated = _sample_paths(cell_rows, cell_cols)
    except RuntimeError:
        return None

    palette = random.sample(PALETTE, len(paths) + 2)
    bg = int(palette[0])
    divider = int(palette[1])
    colors = [int(c) for c in palette[2:]]

    grid, row_segments, col_segments, divider_rows, divider_cols = _blank_grid(
        cell_rows, cell_cols, cell_h, cell_w, bg, divider
    )

    for path in paths:
        for a, b in zip(path, path[1:]):
            _paint_gap(grid, a, b, row_segments, col_segments, divider_rows, divider_cols, bg)

    for path, color in zip(paths, colors):
        _paint_seed(grid, path[0], color, row_segments, col_segments)

    if not _oracle(grid, paths, colors, include_isolated):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid routed-divider grid")
