from __future__ import annotations

import random
from collections import deque
from itertools import combinations

import numpy as np


BG = 0
GREY = 5
STATION_SIZE = 5
INNER_SLICE = slice(1, 4)
N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
TEMP_CANVAS = 30
GEN_RETRIES = 600

STAMP_MASKS = (
    np.array(
        [
            [1, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 1, 1],
            [0, 1, 0],
            [1, 0, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 0, 0],
            [1, 1, 1],
            [0, 0, 1],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 1, 0],
            [1, 1, 0],
            [1, 0, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 0, 1],
            [1, 1, 0],
            [0, 0, 0],
        ],
        dtype=int,
    ),
)


def _neighbors4(r: int, c: int):
    for dr, dc in N4:
        yield r + dr, c + dc


def _line_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    if c0 == c1:
        step = 1 if r1 >= r0 else -1
        return [(r, c0) for r in range(r0, r1 + step, step)]
    raise ValueError("line must be axis-aligned")


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
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


def _frame_cells(top: int, left: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for dc in range(STATION_SIZE):
        cells.append((top, left + dc))
        cells.append((top + STATION_SIZE - 1, left + dc))
    for dr in range(1, STATION_SIZE - 1):
        cells.append((top + dr, left))
        cells.append((top + dr, left + STATION_SIZE - 1))
    return cells


def _draw_station_frame(grid: np.ndarray, top: int, left: int) -> None:
    for r, c in _frame_cells(top, left):
        grid[r, c] = GREY


def _place_stamp(grid: np.ndarray, top: int, left: int, stamp: np.ndarray) -> None:
    grid[top + 1 : top + 4, left + 1 : left + 4] = np.asarray(stamp, dtype=int)


def _detect_stations(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    stations: list[dict[str, object]] = []
    occupied = np.zeros_like(g, dtype=bool)
    for top in range(h - STATION_SIZE + 1):
        for left in range(w - STATION_SIZE + 1):
            patch = g[top : top + STATION_SIZE, left : left + STATION_SIZE]
            if not (
                np.all(patch[0, :] == GREY)
                and np.all(patch[-1, :] == GREY)
                and np.all(patch[:, 0] == GREY)
                and np.all(patch[:, -1] == GREY)
            ):
                continue
            inner = patch[INNER_SLICE, INNER_SLICE]
            if np.any(inner == GREY):
                continue
            mask = np.zeros_like(g, dtype=bool)
            for r, c in _frame_cells(top, left):
                mask[r, c] = True
            if np.any(occupied & mask):
                continue
            occupied |= mask
            stations.append(
                {
                    "top": top,
                    "left": left,
                    "center": (top + 2, left + 2),
                    "inner": inner.copy(),
                }
            )
    return stations


def _frame_mask(shape: tuple[int, int], stations: list[dict[str, object]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for st in stations:
        for r, c in _frame_cells(int(st["top"]), int(st["left"])):
            mask[r, c] = True
    return mask


def _station_ports(st: dict[str, object]) -> dict[str, tuple[int, int]]:
    top = int(st["top"])
    left = int(st["left"])
    return {
        "N": (top - 1, left + 2),
        "E": (top + 2, left + 5),
        "S": (top + 5, left + 2),
        "W": (top + 2, left - 1),
    }


def _build_station_graph(grid: np.ndarray, stations: list[dict[str, object]]) -> dict[int, set[int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    frame = _frame_mask(g.shape, stations)
    wire_mask = (g == GREY) & (~frame)
    port_owner: dict[tuple[int, int], set[int]] = {}
    for idx, st in enumerate(stations):
        for coord in _station_ports(st).values():
            r, c = coord
            if 0 <= r < h and 0 <= c < w and wire_mask[r, c]:
                port_owner.setdefault((r, c), set()).add(idx)
    comps = _components(wire_mask)
    graph = {i: set() for i in range(len(stations))}
    for comp in comps:
        touched = []
        for cell in comp:
            for owner in sorted(port_owner.get(cell, ())):
                if owner not in touched:
                    touched.append(owner)
        assert len(touched) == 2, "each pipe component must connect exactly two stations"
        a, b = touched
        graph[a].add(b)
        graph[b].add(a)
    assert sum(len(v) for v in graph.values()) == 2 * (len(stations) - 1), "network must be a tree"
    return graph


def _bfs_dist(graph: dict[int, set[int]], start: int) -> dict[int, int]:
    dist = {start: 0}
    q = deque([start])
    while q:
        node = q.popleft()
        for nxt in graph[node]:
            if nxt not in dist:
                dist[nxt] = dist[node] + 1
                q.append(nxt)
    return dist


def _nonempty_colors(stamp: np.ndarray) -> set[int]:
    return {int(v) for v in np.unique(stamp) if int(v) != BG}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each empty station from the nearer source stamp, rotating that stamp 90 degrees clockwise per hop.

    Input structure:
    - A black background contains one connected gray tree of 5x5 hollow station frames linked by 1-cell-wide gray pipes.
    - Exactly two stations are sources: their 3x3 interiors contain colored asymmetric stamps; every other station interior is empty black.
    - Pipe lengths and bends vary, but the network is judged by station-to-station hops, not by pixel path length.

    Rule:
    - Parse the station graph and compute, for every empty station, its hop distance to each of the two source stations along the gray pipes.
    - The nearer source supplies the stamp for that station; ties do not occur.
    - Rotate the supplying source stamp clockwise by 90 degrees times the chosen hop count, and draw it into the empty station interior.
    - Keep the gray frames, gray pipes, and both original source stations unchanged.

    Key insight:
    - The same hop count is used twice: first to decide which source owns a station, then to decide the stamp's orientation modulo four.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    assert int(np.min(g)) >= 0 and int(np.max(g)) <= 9

    stations = _detect_stations(g)
    assert len(stations) >= 3
    graph = _build_station_graph(g, stations)
    assert len(graph) == len(stations)

    sources = [idx for idx, st in enumerate(stations) if np.any(np.asarray(st["inner"]) != BG)]
    assert len(sources) == 2, "expected exactly two source stations"
    source_a, source_b = sources
    stamp_a = np.asarray(stations[source_a]["inner"], dtype=int)
    stamp_b = np.asarray(stations[source_b]["inner"], dtype=int)
    assert not np.array_equal(stamp_a, stamp_b)
    assert _nonempty_colors(stamp_a) != _nonempty_colors(stamp_b)

    dist_a = _bfs_dist(graph, source_a)
    dist_b = _bfs_dist(graph, source_b)
    assert len(dist_a) == len(stations) == len(dist_b)

    out = g.copy()
    for idx, st in enumerate(stations):
        inner = np.asarray(st["inner"], dtype=int)
        if np.any(inner != BG):
            continue
        da = dist_a[idx]
        db = dist_b[idx]
        assert da != db, "ties are disallowed by the puzzle"
        if da < db:
            hop = da
            stamp = stamp_a
        else:
            hop = db
            stamp = stamp_b
        rotated = np.rot90(stamp, -int(hop % 4))
        top = int(st["top"])
        left = int(st["left"])
        out[top + 1 : top + 4, left + 1 : left + 4] = rotated
    return out


def _lattice_neighbors(node: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = node
    out = []
    for dr, dc in N4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < 4 and 0 <= nc < 3:
            out.append((nr, nc))
    return out


def _gen_tree() -> tuple[list[tuple[int, int]], dict[tuple[int, int], set[tuple[int, int]]]] | None:
    node_count = random.randint(7, 10)
    start = (random.randrange(4), random.randrange(3))
    nodes = [start]
    node_set = {start}
    graph: dict[tuple[int, int], set[tuple[int, int]]] = {start: set()}
    while len(nodes) < node_count:
        choices = []
        for node in nodes:
            if len(graph[node]) >= 3:
                continue
            for nxt in _lattice_neighbors(node):
                if nxt not in node_set:
                    choices.append((node, nxt))
        if not choices:
            return None
        parent, child = random.choice(choices)
        node_set.add(child)
        nodes.append(child)
        graph.setdefault(child, set()).add(parent)
        graph[parent].add(child)
    if max(len(graph[node]) for node in nodes) < 3:
        return None
    return nodes, graph


def _all_hop_dists(
    nodes: list[tuple[int, int]],
    graph: dict[tuple[int, int], set[tuple[int, int]]],
) -> dict[tuple[int, int], dict[tuple[int, int], int]]:
    dists = {}
    for node in nodes:
        q = deque([node])
        dist = {node: 0}
        while q:
            cur = q.popleft()
            for nxt in graph[cur]:
                if nxt not in dist:
                    dist[nxt] = dist[cur] + 1
                    q.append(nxt)
        dists[node] = dist
    return dists


def _choose_sources(
    nodes: list[tuple[int, int]],
    graph: dict[tuple[int, int], set[tuple[int, int]]],
) -> tuple[tuple[int, int], tuple[int, int], dict[tuple[int, int], dict[tuple[int, int], int]]] | None:
    dists = _all_hop_dists(nodes, graph)
    valid = []
    for a, b in combinations(nodes, 2):
        if dists[a][b] % 2 == 0:
            continue
        territory_a = []
        territory_b = []
        okay = True
        far_hop = 0
        for node in nodes:
            da = dists[a][node]
            db = dists[b][node]
            if da == db:
                okay = False
                break
            if node not in (a, b):
                if da < db:
                    territory_a.append(node)
                    far_hop = max(far_hop, da)
                else:
                    territory_b.append(node)
                    far_hop = max(far_hop, db)
        if not okay:
            continue
        if not territory_a or not territory_b:
            continue
        if far_hop < 4:
            continue
        valid.append((a, b))
    if not valid:
        return None
    src_a, src_b = random.choice(valid)
    return src_a, src_b, dists


def _sample_positions() -> tuple[list[int], list[int]]:
    while True:
        row0 = random.randint(1, 2)
        col0 = random.randint(1, 2)
        row_steps = [random.randint(6, 7) for _ in range(3)]
        col_steps = [random.randint(9, 12) for _ in range(2)]
        rows = [row0]
        cols = [col0]
        for step in row_steps:
            rows.append(rows[-1] + step)
        for step in col_steps:
            cols.append(cols[-1] + step)
        if rows[-1] + STATION_SIZE <= TEMP_CANVAS and cols[-1] + STATION_SIZE <= TEMP_CANVAS:
            return rows, cols


def _path_clear(
    grid: np.ndarray,
    cells: list[tuple[int, int]],
    allowed_touch: set[tuple[int, int]],
) -> bool:
    h, w = grid.shape
    cell_set = set(cells)
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if grid[r, c] != BG:
            return False
    for r, c in cells:
        for nr, nc in _neighbors4(r, c):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if (nr, nc) in cell_set:
                continue
            if grid[nr, nc] == GREY and (nr, nc) not in allowed_touch:
                return False
    return True


def _horizontal_edge_candidates(
    left_top: tuple[int, int],
    right_top: tuple[int, int],
) -> tuple[list[list[tuple[int, int]]], set[tuple[int, int]]]:
    lr, lc = left_top
    rr, rc = right_top
    row = lr + 2
    start = (row, lc + 5)
    end = (row, rc - 1)
    left_port = (lr + 2, lc + 4)
    right_port = (rr + 2, rc)
    gap = end[1] - start[1] + 1
    candidates = [_line_cells(start, end)]
    if gap >= 5:
        shifts = [-2, -1, 1, 2]
        random.shuffle(shifts)
        for shift in shifts:
            bend_row = row + shift
            c1 = start[1] + 1
            c2 = end[1] - 1
            path = _line_cells(start, (row, c1))
            path += _line_cells((row, c1), (bend_row, c1))[1:]
            path += _line_cells((bend_row, c1), (bend_row, c2))[1:]
            path += _line_cells((bend_row, c2), (row, c2))[1:]
            path += _line_cells((row, c2), end)[1:]
            candidates.insert(0, path)
    return candidates, {left_port, right_port}


def _vertical_edge_candidates(
    top_left: tuple[int, int],
    bottom_left: tuple[int, int],
) -> tuple[list[list[tuple[int, int]]], set[tuple[int, int]]]:
    tr, tc = top_left
    br, bc = bottom_left
    col = tc + 2
    start = (tr + 5, col)
    end = (br - 1, col)
    top_port = (tr + 4, tc + 2)
    bottom_port = (br, bc + 2)
    return [_line_cells(start, end)], {top_port, bottom_port}


def _render_edges(
    grid: np.ndarray,
    nodes: list[tuple[int, int]],
    graph: dict[tuple[int, int], set[tuple[int, int]]],
    tops: dict[tuple[int, int], tuple[int, int]],
) -> dict[frozenset[tuple[int, int]], int] | None:
    edge_lengths: dict[frozenset[tuple[int, int]], int] = {}
    done: set[frozenset[tuple[int, int]]] = set()
    for node in nodes:
        for nxt in graph[node]:
            edge = frozenset((node, nxt))
            if edge in done:
                continue
            done.add(edge)
            a = tops[node]
            b = tops[nxt]
            if a[0] == b[0]:
                left, right = (a, b) if a[1] < b[1] else (b, a)
                candidates, allowed = _horizontal_edge_candidates(left, right)
            else:
                top, bottom = (a, b) if a[0] < b[0] else (b, a)
                candidates, allowed = _vertical_edge_candidates(top, bottom)
            chosen = None
            for path in candidates:
                if _path_clear(grid, path, allowed):
                    chosen = path
                    break
            if chosen is None:
                return None
            for r, c in chosen:
                grid[r, c] = GREY
            edge_lengths[edge] = len(chosen)
    return edge_lengths


def _colorize_stamp(mask: np.ndarray, color: int) -> np.ndarray:
    return np.where(mask != 0, color, BG).astype(int)


def _shortest_weighted_lengths(
    nodes: list[tuple[int, int]],
    graph: dict[tuple[int, int], set[tuple[int, int]]],
    edge_lengths: dict[frozenset[tuple[int, int]], int],
    start: tuple[int, int],
) -> dict[tuple[int, int], int]:
    dist = {start: 0}
    q = deque([start])
    while q:
        node = q.popleft()
        for nxt in graph[node]:
            if nxt not in dist:
                dist[nxt] = dist[node] + edge_lengths[frozenset((node, nxt))]
                q.append(nxt)
    return dist


def _make_expected(
    grid: np.ndarray,
    nodes: list[tuple[int, int]],
    tops: dict[tuple[int, int], tuple[int, int]],
    src_a: tuple[int, int],
    src_b: tuple[int, int],
    dists: dict[tuple[int, int], dict[tuple[int, int], int]],
    stamp_a: np.ndarray,
    stamp_b: np.ndarray,
) -> np.ndarray:
    out = grid.copy()
    for node in nodes:
        if node in (src_a, src_b):
            continue
        da = dists[src_a][node]
        db = dists[src_b][node]
        stamp = stamp_a if da < db else stamp_b
        hop = da if da < db else db
        rotated = np.rot90(stamp, -int(hop % 4))
        top, left = tops[node]
        _place_stamp(out, top, left, rotated)
    return out


def _transform_pair(inp: np.ndarray, out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    k = random.randrange(4)
    inp = np.rot90(inp, k)
    out = np.rot90(out, k)
    if random.random() < 0.5:
        inp = np.fliplr(inp)
        out = np.fliplr(out)
    used = (inp != BG) | (out != BG)
    coords = np.argwhere(used)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    pad_top = random.randint(0, 2)
    pad_bottom = random.randint(0, 2)
    pad_left = random.randint(0, 2)
    pad_right = random.randint(0, 2)
    r0 = max(0, int(r0) - pad_top)
    c0 = max(0, int(c0) - pad_left)
    r1 = min(inp.shape[0] - 1, int(r1) + pad_bottom)
    c1 = min(inp.shape[1] - 1, int(c1) + pad_right)
    return inp[r0 : r1 + 1, c0 : c1 + 1].copy(), out[r0 : r1 + 1, c0 : c1 + 1].copy()


def _semantic_checks(
    inp: np.ndarray,
    expected: np.ndarray,
    nodes: list[tuple[int, int]],
    graph: dict[tuple[int, int], set[tuple[int, int]]],
    src_a: tuple[int, int],
    src_b: tuple[int, int],
    dists: dict[tuple[int, int], dict[tuple[int, int], int]],
    edge_lengths: dict[frozenset[tuple[int, int]], int],
) -> bool:
    if not np.array_equal(main(inp), expected):
        return False

    weighted_a = _shortest_weighted_lengths(nodes, graph, edge_lengths, src_a)
    weighted_b = _shortest_weighted_lengths(nodes, graph, edge_lengths, src_b)

    has_far = False
    has_rotation = False
    has_weight_conflict = False
    claims_a = 0
    claims_b = 0
    for node in nodes:
        if node in (src_a, src_b):
            continue
        da = dists[src_a][node]
        db = dists[src_b][node]
        near_hop = 0 if da < db else 1
        near_weight = 0 if weighted_a[node] < weighted_b[node] else 1
        hop = min(da, db)
        if da < db:
            claims_a += 1
        else:
            claims_b += 1
        if hop >= 4:
            has_far = True
        if hop % 4 != 0:
            has_rotation = True
        if near_hop != near_weight:
            has_weight_conflict = True
    if claims_a == 0 or claims_b == 0:
        return False
    if not has_far or not has_rotation or not has_weight_conflict:
        return False

    parsed_stations = _detect_stations(inp)
    if len(parsed_stations) != len(nodes):
        return False
    parsed_graph = _build_station_graph(inp, parsed_stations)
    if sum(len(v) for v in parsed_graph.values()) != 2 * (len(nodes) - 1):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    tree = _gen_tree()
    if tree is None:
        return None
    nodes, graph = tree
    picked = _choose_sources(nodes, graph)
    if picked is None:
        return None
    src_a, src_b, dists = picked

    rows, cols = _sample_positions()
    tops = {(r, c): (rows[r], cols[c]) for r, c in nodes}

    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], 2)
    mask_a, mask_b = random.sample(STAMP_MASKS, 2)
    stamp_a = _colorize_stamp(mask_a, colors[0])
    stamp_b = _colorize_stamp(mask_b, colors[1])

    grid = np.zeros((TEMP_CANVAS, TEMP_CANVAS), dtype=int)
    for node in nodes:
        top, left = tops[node]
        _draw_station_frame(grid, top, left)
    edge_lengths = _render_edges(grid, nodes, graph, tops)
    if edge_lengths is None:
        return None

    top_a, left_a = tops[src_a]
    top_b, left_b = tops[src_b]
    _place_stamp(grid, top_a, left_a, stamp_a)
    _place_stamp(grid, top_b, left_b, stamp_b)

    expected = _make_expected(grid, nodes, tops, src_a, src_b, dists, stamp_a, stamp_b)
    inp, out = _transform_pair(grid, expected)
    if not _semantic_checks(inp, out, nodes, graph, src_a, src_b, dists, edge_lengths):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid dual-source pipeline puzzle")
