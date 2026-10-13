from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _infer_main_bg(grid: np.ndarray, sep: int, wall: int) -> int:
    vals = [int(v) for v in grid[:sep].ravel() if int(v) != wall]
    assert vals, "main area must contain non-wall cells"
    return int(Counter(vals).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    h, w = mask.shape
    labels = np.full((h, w), -1, dtype=int)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != -1:
                continue
            cid = len(comps)
            labels[r, c] = cid
            q = deque([(r, c)])
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == -1:
                        labels[nr, nc] = cid
                        q.append((nr, nc))
            comps.append(cells)
    return labels, comps


def _bfs_dist(room_count: int, edges: set[tuple[int, int]], source: int) -> dict[int, int]:
    adj = {i: set() for i in range(room_count)}
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    dist = {source: 0}
    q = deque([source])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v in dist:
                continue
            dist[v] = dist[u] + 1
            q.append(v)
    return dist


def _find_separator_and_wall(grid: np.ndarray) -> tuple[int, int]:
    candidates: list[int] = []
    for r in range(grid.shape[0] - 1):
        if np.all(grid[r] == grid[r, 0]) and np.any(grid[r + 1 :] != grid[r, 0]):
            candidates.append(r)
    assert candidates, "expected a full-width separator wall above the legend"
    sep = max(candidates)
    return int(sep), int(grid[sep, 0])


def _extract_legend(grid: np.ndarray, sep: int, bg: int, wall: int) -> list[int]:
    assert sep + 1 < grid.shape[0], "expected legend rows below the separator"
    best_row = None
    best_count = -1
    for r in range(sep + 1, grid.shape[0]):
        count = int(np.count_nonzero((grid[r] != bg) & (grid[r] != wall)))
        if count > best_count:
            best_count = count
            best_row = r
    assert best_row is not None and best_count > 0, "expected a visible legend row"
    cols = np.flatnonzero((grid[best_row] != bg) & (grid[best_row] != wall))
    assert cols.size > 0, "legend row must contain colored cells"
    assert np.array_equal(cols, np.arange(int(cols[0]), int(cols[-1]) + 1)), "legend must be contiguous"
    return [int(v) for v in grid[best_row, cols]]


def _cell_orientation(grid: np.ndarray, r: int, c: int, wall: int, sep: int) -> str | None:
    if not (1 <= r < sep and 1 <= c < grid.shape[1] - 1):
        return None
    vertical = (
        int(grid[r - 1, c]) == wall
        and int(grid[r + 1, c]) == wall
        and int(grid[r, c - 1]) != wall
        and int(grid[r, c + 1]) != wall
    )
    horizontal = (
        int(grid[r, c - 1]) == wall
        and int(grid[r, c + 1]) == wall
        and int(grid[r - 1, c]) != wall
        and int(grid[r + 1, c]) != wall
    )
    assert not (vertical and horizontal), "wall gap cannot have two orientations"
    if vertical:
        return "v"
    if horizontal:
        return "h"
    return None


def _analyze_structure(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep, wall = _find_separator_and_wall(g)
    bg = _infer_main_bg(g, sep, wall)
    legend = _extract_legend(g, sep, bg, wall)

    door_cells: dict[tuple[int, int], str] = {}
    blocker_cells: dict[tuple[int, int], str] = {}
    for r in range(1, sep):
        for c in range(1, g.shape[1] - 1):
            orient = _cell_orientation(g, r, c, wall, sep)
            if orient is None:
                continue
            value = int(g[r, c])
            if value == bg:
                door_cells[(r, c)] = orient
            elif value != wall:
                blocker_cells[(r, c)] = orient

    source_candidates = []
    for r in range(sep):
        for c in range(g.shape[1]):
            value = int(g[r, c])
            if value in (bg, wall):
                continue
            if (r, c) in blocker_cells:
                continue
            source_candidates.append((r, c))
    assert len(source_candidates) == 1, "expected exactly one source marker"
    source = source_candidates[0]

    traversable = np.zeros_like(g, dtype=bool)
    traversable[:sep] = (g[:sep] == bg)
    traversable[source] = True
    for door in door_cells:
        traversable[door] = False
    for blocker in blocker_cells:
        traversable[blocker] = False

    room_ids, rooms = _components(traversable)
    assert rooms, "expected at least one room"
    source_room = int(room_ids[source])
    assert source_room >= 0, "source must lie inside a room"

    edges: set[tuple[int, int]] = set()
    for (r, c), orient in door_cells.items():
        if orient == "v":
            a, b = int(room_ids[r, c - 1]), int(room_ids[r, c + 1])
        else:
            a, b = int(room_ids[r - 1, c]), int(room_ids[r + 1, c])
        assert a >= 0 and b >= 0 and a != b, "door must connect two distinct rooms"
        edges.add((min(a, b), max(a, b)))

    dist = _bfs_dist(len(rooms), edges, source_room)
    assert len(dist) == len(rooms), "all rooms must be reachable from the source room"
    assert max(dist.values()) + 1 == len(legend), "legend length must match the maximum door distance"

    return {
        "grid": g,
        "bg": int(bg),
        "wall": int(wall),
        "sep": int(sep),
        "legend": legend,
        "doors": door_cells,
        "blockers": blocker_cells,
        "source": source,
        "source_room": int(source_room),
        "room_ids": room_ids,
        "rooms": rooms,
        "dist": dist,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Color each room by its minimum number of crossed doors from the source room.

    Input structure:
    - Grey-like wall pixels partition the upper map into enclosed rooms.
    - A one-cell gap in a wall is a true door only when the gap cell is still background; a colored pixel plugging such a gap is a fake blocked door.
    - One room contains a singleton source marker, and a bottom legend strip isolated by a full wall row lists the layer colors from left to right.

    Rule:
    - Treat every room as a graph node and every true door as an undirected edge.
    - Compute each room's shortest door-crossing distance d from the source room.
    - Recolor every background cell inside that room with the legend's (d+1)-th color.
    - Preserve walls, fake-door blocker pixels, the legend strip, the source marker, and leave each door-gap cell as background.

    Key insight:
    - The coloring layers come from shortest path distance in the room-adjacency graph, not from geometric closeness or flood-fill radius in the pixel grid.
    """
    info = _analyze_structure(grid)
    out = np.array(info["grid"], copy=True)
    bg = int(info["bg"])
    legend = info["legend"]
    room_ids = info["room_ids"]

    for rid, cells in enumerate(info["rooms"]):
        color = int(legend[info["dist"][rid]])
        for r, c in cells:
            if int(out[r, c]) == bg:
                out[r, c] = color
    return out


def _room_boxes(row_heights: list[int], col_widths: list[int]) -> dict[int, tuple[int, int, int, int]]:
    boxes: dict[int, tuple[int, int, int, int]] = {}
    row_start = 1
    room_cols = len(col_widths)
    for i, height in enumerate(row_heights):
        col_start = 1
        for j, width in enumerate(col_widths):
            rid = i * room_cols + j
            boxes[rid] = (row_start, col_start, row_start + height - 1, col_start + width - 1)
            col_start += width + 1
        row_start += height + 1
    return boxes


def _candidate_edges(room_rows: int, room_cols: int):
    edges = []
    for i in range(room_rows):
        for j in range(room_cols):
            a = i * room_cols + j
            if j + 1 < room_cols:
                b = i * room_cols + (j + 1)
                edges.append({"pair": (a, b), "orient": "v", "grid_pos": (i, j)})
            if i + 1 < room_rows:
                b = (i + 1) * room_cols + j
                edges.append({"pair": (a, b), "orient": "h", "grid_pos": (i, j)})
    return edges


def _boundary_cells(edge: dict, boxes: dict[int, tuple[int, int, int, int]], room_cols: int) -> list[tuple[int, int]]:
    a, _b = edge["pair"]
    i, j = divmod(a, room_cols)
    if edge["orient"] == "v":
        r0, _c0, r1, c1 = boxes[a]
        return [(r, c1 + 1) for r in range(r0, r1 + 1)]
    _r0, c0, r1, c1 = boxes[a]
    return [(r1 + 1, c) for c in range(c0, c1 + 1)]


def _random_connected_edges(room_rows: int, room_cols: int):
    nodes = list(range(room_rows * room_cols))
    candidates = _candidate_edges(room_rows, room_cols)
    by_node: dict[int, list[int]] = {n: [] for n in nodes}
    for idx, edge in enumerate(candidates):
        a, b = edge["pair"]
        by_node[a].append(idx)
        by_node[b].append(idx)

    start = random.choice(nodes)
    visited = {start}
    chosen: set[int] = set()
    while len(visited) < len(nodes):
        frontier = []
        for idx, edge in enumerate(candidates):
            if idx in chosen:
                continue
            a, b = edge["pair"]
            if (a in visited) ^ (b in visited):
                frontier.append(idx)
        assert frontier, "grid adjacency graph must support a spanning tree"
        idx = random.choice(frontier)
        chosen.add(idx)
        a, b = candidates[idx]["pair"]
        visited.add(a)
        visited.add(b)

    unused = [idx for idx in range(len(candidates)) if idx not in chosen]
    random.shuffle(unused)
    extra_cap = min(3, len(unused))
    extra_count = random.randint(0, extra_cap)
    for idx in unused[:extra_count]:
        chosen.add(idx)

    true_edges = {tuple(sorted(candidates[idx]["pair"])) for idx in chosen}
    return candidates, true_edges


def _changed_by_opening(room_count: int, true_edges: set[tuple[int, int]], source: int, candidate: tuple[int, int]) -> bool:
    old = _bfs_dist(room_count, true_edges, source)
    new = _bfs_dist(room_count, true_edges | {candidate}, source)
    return old != new


def _pick_graph(room_rows: int, room_cols: int):
    room_count = room_rows * room_cols
    for _ in range(400):
        candidates, true_edges = _random_connected_edges(room_rows, room_cols)
        all_pairs = [tuple(sorted(edge["pair"])) for edge in candidates]
        non_edges = [pair for pair in all_pairs if pair not in true_edges]
        if not non_edges:
            continue

        by_source: list[tuple[int, dict[int, int], list[tuple[int, int]]]] = []
        for source in range(room_count):
            dist = _bfs_dist(room_count, true_edges, source)
            if len(dist) != room_count:
                continue
            maxdist = max(dist.values())
            if not (2 <= maxdist <= 5):
                continue
            incident_non_edges = [pair for pair in non_edges if source in pair]
            if not incident_non_edges:
                continue
            by_source.append((source, dist, incident_non_edges))
        if not by_source:
            continue

        source, dist, incident_non_edges = random.choice(by_source)
        critical_incident = [pair for pair in incident_non_edges if _changed_by_opening(room_count, true_edges, source, pair)]
        if not critical_incident:
            continue

        chosen_blockers = [random.choice(critical_incident)]
        other_non_edges = [pair for pair in non_edges if pair not in chosen_blockers]
        random.shuffle(other_non_edges)
        blocker_target = random.randint(1, min(3, len(non_edges)))
        for pair in other_non_edges:
            if len(chosen_blockers) >= blocker_target:
                break
            if pair not in chosen_blockers:
                chosen_blockers.append(pair)

        if not any(_changed_by_opening(room_count, true_edges, source, pair) for pair in chosen_blockers):
            continue

        return candidates, true_edges, source, dist, set(chosen_blockers)
    raise RuntimeError("failed to construct a valid room graph")


def _sample_colors(layer_count: int) -> tuple[int, int, list[int], int]:
    palette = list(range(10))
    bg = random.choice(palette)
    palette.remove(bg)
    wall = random.choice(palette)
    palette.remove(wall)
    legend = random.sample(palette, layer_count)
    for color in legend:
        palette.remove(color)
    noise = random.choice(palette)
    return int(bg), int(wall), [int(c) for c in legend], int(noise)


def _render_input(
    row_heights: list[int],
    col_widths: list[int],
    true_edges: set[tuple[int, int]],
    blocker_edges: set[tuple[int, int]],
    source_room: int,
    legend: list[int],
    bg: int,
    wall: int,
    noise: int,
) -> tuple[np.ndarray, dict]:
    room_rows = len(row_heights)
    room_cols = len(col_widths)
    boxes = _room_boxes(row_heights, col_widths)
    main_h = sum(row_heights) + room_rows + 1
    width = sum(col_widths) + room_cols + 1
    height = main_h + 1
    grid = np.full((height, width), bg, dtype=int)
    grid[:main_h, :] = wall

    for rid, (r0, c0, r1, c1) in boxes.items():
        grid[r0 : r1 + 1, c0 : c1 + 1] = bg

    candidates = _candidate_edges(room_rows, room_cols)
    chosen_doors: dict[tuple[int, int], tuple[int, int]] = {}
    chosen_blockers: dict[tuple[int, int], tuple[int, int]] = {}
    for edge in candidates:
        pair = tuple(sorted(edge["pair"]))
        cells = _boundary_cells(edge, boxes, room_cols)
        cell = random.choice(cells)
        if pair in true_edges:
            grid[cell] = bg
            chosen_doors[pair] = cell
        elif pair in blocker_edges:
            grid[cell] = noise
            chosen_blockers[pair] = cell

    sr0, sc0, sr1, sc1 = boxes[source_room]
    source = (random.randint(sr0, sr1), random.randint(sc0, sc1))
    grid[source] = legend[0]

    start_col = random.randint(0, width - len(legend))
    grid[-1, start_col : start_col + len(legend)] = np.array(legend, dtype=int)

    meta = {
        "boxes": boxes,
        "doors": chosen_doors,
        "blockers": chosen_blockers,
        "source": source,
        "source_room": source_room,
    }
    return grid, meta


def _expected_from_spec(
    grid: np.ndarray,
    boxes: dict[int, tuple[int, int, int, int]],
    dist: dict[int, int],
    legend: list[int],
    bg: int,
) -> np.ndarray:
    out = np.array(grid, copy=True)
    for rid, (r0, c0, r1, c1) in boxes.items():
        color = legend[dist[rid]]
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        patch[patch == bg] = color
    return out


def _validate_generated(
    grid: np.ndarray,
    expected: np.ndarray,
    true_edges: set[tuple[int, int]],
    blocker_edges: set[tuple[int, int]],
    source_room: int,
    dist: dict[int, int],
    legend: list[int],
    meta: dict,
) -> bool:
    if not np.array_equal(main(grid), expected):
        return False
    if np.array_equal(grid, expected):
        return False
    if max(dist.values()) < 2:
        return False
    if len(set(legend)) != len(legend):
        return False
    if not blocker_edges:
        return False
    if not any(source_room in edge for edge in blocker_edges):
        return False

    room_count = len(meta["boxes"])
    if not any(_changed_by_opening(room_count, true_edges, source_room, edge) for edge in blocker_edges):
        return False

    info = _analyze_structure(grid)
    if len(info["blockers"]) != len(blocker_edges):
        return False
    if len(info["doors"]) != len(true_edges):
        return False
    if [int(v) for v in info["legend"]] != [int(v) for v in legend]:
        return False
    if max(info["dist"].values()) != max(dist.values()):
        return False
    if any(expected[cell] != _most_common_color(grid) for cell in meta["doors"].values()):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(500):
        room_rows = random.randint(2, 4)
        room_cols = random.randint(2, 4)
        row_heights = [random.randint(2, 4) for _ in range(room_rows)]
        col_widths = [random.randint(2, 5) for _ in range(room_cols)]

        main_h = sum(row_heights) + room_rows + 1
        width = sum(col_widths) + room_cols + 1
        if max(main_h + 1, width) > 30:
            continue

        try:
            candidates, true_edges, source_room, dist, blocker_edges = _pick_graph(room_rows, room_cols)
        except RuntimeError:
            continue
        maxdist = max(dist.values())
        bg, wall, legend, noise = _sample_colors(maxdist + 1)

        grid, meta = _render_input(
            row_heights=row_heights,
            col_widths=col_widths,
            true_edges=true_edges,
            blocker_edges=blocker_edges,
            source_room=source_room,
            legend=legend,
            bg=bg,
            wall=wall,
            noise=noise,
        )
        expected = _expected_from_spec(grid, meta["boxes"], dist, legend, bg)
        if _validate_generated(grid, expected, true_edges, blocker_edges, source_room, dist, legend, meta):
            return grid
    raise RuntimeError("failed to generate a valid layered-door-distance puzzle")
