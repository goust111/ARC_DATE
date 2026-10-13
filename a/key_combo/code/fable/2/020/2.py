from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


MOVER_COLOR = 4
CARDINAL_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
LEGEND_SIZE = 5
LEGEND_CENTER_CELLS = ((2, 2), (1, 2), (2, 1), (2, 3), (3, 2))
ROOM_COUNT_RANGE = (3, 6)
GRID_HEIGHT_RANGE = (18, 26)
GRID_WIDTH_RANGE = (18, 26)
MIN_ROOM_SPAN = 4
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _label_mask(mask: np.ndarray) -> tuple[np.ndarray, int]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    current = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            current += 1
            labels[r, c] = current
            queue = deque([(r, c)])
            while queue:
                rr, cc = queue.popleft()
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = current
                        queue.append((nr, nc))
    return labels, current


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _sign(value: float) -> int:
    if abs(value) < 1e-9:
        return 0
    return 1 if value > 0 else -1


def _opposite(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return -dr, -dc


def _segments_for_direction(
    cells: frozenset[tuple[int, int]] | set[tuple[int, int]],
    direction: tuple[int, int],
) -> list[list[tuple[int, int]]]:
    dr, dc = direction
    lanes: dict[int, list[int]] = {}
    if dc != 0:
        for r, c in cells:
            lanes.setdefault(int(r), []).append(int(c))
        segments: list[list[tuple[int, int]]] = []
        for r, cols in lanes.items():
            cols.sort()
            run = [(r, cols[0])]
            for col in cols[1:]:
                if col == run[-1][1] + 1:
                    run.append((r, col))
                else:
                    segments.append(run)
                    run = [(r, col)]
            segments.append(run)
        return segments

    for r, c in cells:
        lanes.setdefault(int(c), []).append(int(r))
    segments = []
    for c, rows in lanes.items():
        rows.sort()
        run = [(rows[0], c)]
        for row in rows[1:]:
            if row == run[-1][0] + 1:
                run.append((row, c))
            else:
                segments.append(run)
                run = [(row, c)]
        segments.append(run)
    return segments


def _stack_segmentwise(
    cells: frozenset[tuple[int, int]] | set[tuple[int, int]],
    points: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    direction: tuple[int, int],
) -> list[tuple[int, int]]:
    point_set = set(points)
    dr, dc = direction
    use_tail = dr > 0 or dc > 0
    final: list[tuple[int, int]] = []
    for segment in _segments_for_direction(cells, direction):
        count = sum(1 for cell in segment if cell in point_set)
        if count == 0:
            continue
        final.extend(segment[-count:] if use_tail else segment[:count])
    return final


def _slide_points_stepwise(
    cells: frozenset[tuple[int, int]] | set[tuple[int, int]],
    points: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    direction: tuple[int, int],
) -> list[tuple[int, int]]:
    dr, dc = direction
    ordered = sorted(points, key=lambda p: (dr * p[0] + dc * p[1], p[0], p[1]), reverse=True)
    occupied: set[tuple[int, int]] = set()
    final: list[tuple[int, int]] = []
    for r0, c0 in ordered:
        r, c = int(r0), int(c0)
        while (r + dr, c + dc) in cells and (r + dr, c + dc) not in occupied:
            r += dr
            c += dc
        occupied.add((r, c))
        final.append((r, c))
    return final


def _parse_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    bg = _get_bg(g)
    colors = sorted(int(color) for color in np.unique(g))
    assert MOVER_COLOR in colors, "yellow movable pixels must be present"

    other_colors = [color for color in colors if color not in (bg, MOVER_COLOR)]
    assert len(other_colors) == 4, "expected one wall color plus three legend colors"
    counts = {color: int(np.sum(g == color)) for color in other_colors}
    wall_color = max(other_colors, key=lambda color: counts[color])
    legend_colors = [color for color in other_colors if color != wall_color]

    legend_mask = np.isin(g, legend_colors)
    legend_bbox = _bbox_from_mask(legend_mask)
    assert legend_bbox is not None, "legend must exist"
    lr0, lc0, lr1, lc1 = legend_bbox
    assert lr0 == 0 or lc0 == 0 or lr1 == h - 1 or lc1 == w - 1, "legend must touch the border"
    assert np.all(legend_mask[lr0 : lr1 + 1, lc0 : lc1 + 1]), "legend must be a filled rectangle"

    legend = g[lr0 : lr1 + 1, lc0 : lc1 + 1]
    legend_counts = {color: int(np.sum(legend == color)) for color in legend_colors}
    fill_color = max(legend_colors, key=lambda color: legend_counts[color])
    remaining = [color for color in legend_colors if color != fill_color]
    center_color = max(remaining, key=lambda color: legend_counts[color])
    indicator_color = min(remaining, key=lambda color: legend_counts[color])
    assert legend_counts[center_color] > legend_counts[indicator_color], "legend center must be distinguishable"

    center_pts = np.argwhere(legend == center_color)
    indicator_pts = np.argwhere(legend == indicator_color)
    delta = indicator_pts.mean(axis=0) - center_pts.mean(axis=0)
    anchor_dir = (_sign(float(delta[0])), _sign(float(delta[1])))
    assert abs(anchor_dir[0]) + abs(anchor_dir[1]) == 1, "generator uses cardinal legend directions"

    passable = (g == bg) | (g == MOVER_COLOR)
    labels, n_regions = _label_mask(passable)
    assert n_regions >= 2, "walls must partition the space into multiple enclosed regions"

    regions: dict[int, dict] = {}
    for region_id in range(1, n_regions + 1):
        coords = np.argwhere(labels == region_id)
        cells = frozenset((int(r), int(c)) for r, c in coords)
        assert cells
        assert not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in cells), "regions must be enclosed"
        movers = tuple((r, c) for r, c in cells if int(g[r, c]) == MOVER_COLOR)
        regions[region_id] = {"cells": cells, "movers": movers}

    anchor_candidates: set[int] = set()
    for r, c in np.argwhere(legend_mask):
        for nr, nc in _neighbors4(int(r), int(c)):
            if 0 <= nr < h and 0 <= nc < w and labels[nr, nc] > 0:
                anchor_candidates.add(int(labels[nr, nc]))
    assert len(anchor_candidates) == 1, "legend must touch exactly one region"
    anchor_region = next(iter(anchor_candidates))

    graph = {region_id: set() for region_id in regions}
    for r, c in np.argwhere(g == wall_color):
        r = int(r)
        c = int(c)
        if 0 < r < h - 1:
            a = int(labels[r - 1, c])
            b = int(labels[r + 1, c])
            if a > 0 and b > 0 and a != b:
                graph[a].add(b)
                graph[b].add(a)
        if 0 < c < w - 1:
            a = int(labels[r, c - 1])
            b = int(labels[r, c + 1])
            if a > 0 and b > 0 and a != b:
                graph[a].add(b)
                graph[b].add(a)

    parity = {anchor_region: 0}
    distance = {anchor_region: 0}
    queue = deque([anchor_region])
    while queue:
        region_id = queue.popleft()
        for neighbor in graph[region_id]:
            wanted = 1 - parity[region_id]
            if neighbor in parity:
                assert parity[neighbor] == wanted, "region adjacency must be bipartite"
                continue
            parity[neighbor] = wanted
            distance[neighbor] = distance[region_id] + 1
            queue.append(neighbor)
    assert len(parity) == n_regions, "all regions must connect through shared walls"

    directions = {
        region_id: anchor_dir if parity[region_id] == 0 else _opposite(anchor_dir)
        for region_id in regions
    }
    return {
        "grid": g,
        "bg": bg,
        "wall_color": wall_color,
        "legend_mask": legend_mask,
        "anchor_dir": anchor_dir,
        "anchor_region": anchor_region,
        "regions": regions,
        "graph": graph,
        "distance": distance,
        "directions": directions,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide yellow pixels inside wall-separated rooms, with the legend-adjacent room fixing the alternating direction pattern.

    Input structure:
    - The background is the most common color, and one-cell-thick walls partition the grid into enclosed passable regions.
    - Yellow movable pixels are scattered inside those regions.
    - Exactly one small legend rectangle sits on the border; its center mark and indicator mark encode one cardinal slide direction, and the legend touches exactly one region.

    Rule:
    - The region touching the legend uses the legend direction.
    - Whenever two regions share a wall, their directions are opposite, so the legend direction propagates across the region-adjacency graph by parity.
    - Inside each region, all yellow pixels slide in that region's direction until blocked by the wall boundary or by already packed yellow pixels in the same lane.
    - Walls and the legend remain unchanged.

    Key insight:
    - The legend does not control the whole grid directly; it only anchors one side of a bipartite region graph, and direction flips once per crossed wall.
    """
    parsed = _parse_grid(grid)
    g = parsed["grid"]
    out = g.copy()
    out[out == MOVER_COLOR] = parsed["bg"]

    for region_id, info in parsed["regions"].items():
        final = _stack_segmentwise(info["cells"], info["movers"], parsed["directions"][region_id])
        for r, c in final:
            out[r, c] = MOVER_COLOR
    return out


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def _room_graph(rooms: dict[int, tuple[int, int, int, int]]) -> dict[int, set[int]]:
    ids = sorted(rooms)
    graph = {room_id: set() for room_id in ids}
    for i, a in enumerate(ids):
        ar0, ac0, ar1, ac1 = rooms[a]
        for b in ids[i + 1 :]:
            br0, bc0, br1, bc1 = rooms[b]
            touch_h = ar1 + 2 == br0 or br1 + 2 == ar0
            touch_v = ac1 + 2 == bc0 or bc1 + 2 == ac0
            if touch_h and _intervals_overlap(ac0, ac1, bc0, bc1):
                graph[a].add(b)
                graph[b].add(a)
            if touch_v and _intervals_overlap(ar0, ar1, br0, br1):
                graph[a].add(b)
                graph[b].add(a)
    return graph


def _bfs_distances(graph: dict[int, set[int]], start: int) -> dict[int, int]:
    dist = {start: 0}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if neighbor not in dist:
                dist[neighbor] = dist[node] + 1
                queue.append(neighbor)
    return dist


def _legend_sites(
    rect: tuple[int, int, int, int],
    height: int,
    width: int,
) -> list[tuple[str, int]]:
    r0, c0, r1, c1 = rect
    room_h = r1 - r0 + 1
    room_w = c1 - c0 + 1
    sites: list[tuple[str, int]] = []
    if r0 == 1 and room_h >= 7 and room_w >= 7:
        for start in range(c0 + 1, c1 - LEGEND_SIZE + 1):
            sites.append(("top", start))
    if r1 == height - 2 and room_h >= 7 and room_w >= 7:
        for start in range(c0 + 1, c1 - LEGEND_SIZE + 1):
            sites.append(("bottom", start))
    if c0 == 1 and room_h >= 7 and room_w >= 7:
        for start in range(r0 + 1, r1 - LEGEND_SIZE + 1):
            sites.append(("left", start))
    if c1 == width - 2 and room_h >= 7 and room_w >= 7:
        for start in range(r0 + 1, r1 - LEGEND_SIZE + 1):
            sites.append(("right", start))
    return sites


def _legend_patch(fill: int, center: int, indicator: int, direction: tuple[int, int]) -> np.ndarray:
    patch = np.full((LEGEND_SIZE, LEGEND_SIZE), int(fill), dtype=int)
    for r, c in LEGEND_CENTER_CELLS:
        patch[r, c] = int(center)
    dr, dc = direction
    patch[2 + 2 * dr, 2 + 2 * dc] = int(indicator)
    return patch


def _paint_legend(
    grid: np.ndarray,
    rect: tuple[int, int, int, int],
    site: tuple[str, int],
    fill: int,
    center: int,
    indicator: int,
    direction: tuple[int, int],
) -> set[tuple[int, int]]:
    h, w = grid.shape
    side, start = site
    if side == "top":
        r0, c0 = 0, start
    elif side == "bottom":
        r0, c0 = h - LEGEND_SIZE, start
    elif side == "left":
        r0, c0 = start, 0
    else:
        r0, c0 = start, w - LEGEND_SIZE

    patch = _legend_patch(fill, center, indicator, direction)
    grid[r0 : r0 + LEGEND_SIZE, c0 : c0 + LEGEND_SIZE] = patch
    return {
        (int(r), int(c))
        for r in range(r0, r0 + LEGEND_SIZE)
        for c in range(c0, c0 + LEGEND_SIZE)
    }


def _room_cells_from_layout(
    rooms: dict[int, tuple[int, int, int, int]],
    anchor_room: int,
    legend_cells: set[tuple[int, int]],
) -> dict[int, frozenset[tuple[int, int]]]:
    room_cells: dict[int, frozenset[tuple[int, int]]] = {}
    for room_id, (r0, c0, r1, c1) in rooms.items():
        cells = {
            (r, c)
            for r in range(r0, r1 + 1)
            for c in range(c0, c1 + 1)
            if room_id != anchor_room or (r, c) not in legend_cells
        }
        room_cells[room_id] = frozenset(cells)
    return room_cells


def _cells_connected(cells: frozenset[tuple[int, int]]) -> bool:
    start = next(iter(cells))
    seen = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return len(seen) == len(cells)


def _pick_initial_positions(
    segment: list[tuple[int, int]],
    direction: tuple[int, int],
    count: int,
) -> tuple[tuple[int, int], ...]:
    dr, dc = direction
    use_tail = dr > 0 or dc > 0
    pool = segment[:-count] if use_tail else segment[count:]
    chosen = random.sample(pool, count)
    chosen.sort()
    return tuple(chosen)


def _choose_movers(
    room_cells: dict[int, frozenset[tuple[int, int]]],
    directions: dict[int, tuple[int, int]],
) -> dict[int, tuple[tuple[int, int], ...]] | None:
    usable_segments: dict[int, list[list[tuple[int, int]]]] = {}
    max_caps: dict[int, int] = {}
    multi_candidates: list[int] = []

    for room_id, cells in room_cells.items():
        segments = [segment for segment in _segments_for_direction(cells, directions[room_id]) if len(segment) >= 2]
        if not segments:
            return None
        usable_segments[room_id] = segments
        cap = max(min(3, len(segment) // 2) for segment in segments)
        if cap < 1:
            return None
        max_caps[room_id] = cap
        if cap >= 2:
            multi_candidates.append(room_id)

    if not multi_candidates:
        return None
    forced_multi = random.choice(multi_candidates)

    movers: dict[int, tuple[tuple[int, int], ...]] = {}
    for room_id, segments in usable_segments.items():
        cap = max_caps[room_id]
        if room_id == forced_multi:
            count = random.randint(2, cap)
        elif cap >= 2 and random.random() < 0.35:
            count = random.randint(1, min(2, cap))
        else:
            count = 1
        candidates = [segment for segment in segments if len(segment) // 2 >= count]
        if not candidates:
            return None
        segment = random.choice(candidates)
        movers[room_id] = _pick_initial_positions(segment, directions[room_id], count)
    return movers


def _simulate_known_layout(
    grid: np.ndarray,
    bg: int,
    room_cells: dict[int, frozenset[tuple[int, int]]],
    directions: dict[int, tuple[int, int]],
    movers: dict[int, tuple[tuple[int, int], ...]],
) -> np.ndarray:
    out = grid.copy()
    out[out == MOVER_COLOR] = int(bg)
    for room_id, cells in room_cells.items():
        final = _slide_points_stepwise(cells, movers[room_id], directions[room_id])
        for r, c in final:
            out[r, c] = MOVER_COLOR
    return out


def _validate_generated(
    grid: np.ndarray,
    bg: int,
    room_cells: dict[int, frozenset[tuple[int, int]]],
    graph: dict[int, set[int]],
    anchor_room: int,
    anchor_dir: tuple[int, int],
    directions: dict[int, tuple[int, int]],
    movers: dict[int, tuple[tuple[int, int], ...]],
) -> bool:
    if _get_bg(grid) != bg:
        return False
    if not all(movers[room_id] for room_id in room_cells):
        return False
    if not any(len(movers[room_id]) >= 2 for room_id in room_cells):
        return False
    if len({directions[room_id] for room_id in room_cells}) != 2:
        return False

    parsed = _parse_grid(grid)
    meta_by_cells = {cells: room_id for room_id, cells in room_cells.items()}
    parsed_by_cells = {info["cells"]: region_id for region_id, info in parsed["regions"].items()}
    if set(meta_by_cells) != set(parsed_by_cells):
        return False

    room_to_region = {
        room_id: parsed_by_cells[cells]
        for cells, room_id in meta_by_cells.items()
    }
    if room_to_region[anchor_room] != parsed["anchor_region"]:
        return False
    if parsed["anchor_dir"] != anchor_dir:
        return False

    meta_edges = {
        tuple(sorted((room_to_region[a], room_to_region[b])))
        for a, neighbors in graph.items()
        for b in neighbors
        if a < b
    }
    parsed_edges = {
        tuple(sorted((a, b)))
        for a, neighbors in parsed["graph"].items()
        for b in neighbors
        if a < b
    }
    if meta_edges != parsed_edges:
        return False

    for room_id, region_id in room_to_region.items():
        if parsed["directions"][region_id] != directions[room_id]:
            return False
        if set(parsed["regions"][region_id]["movers"]) != set(movers[room_id]):
            return False

    expected = _simulate_known_layout(grid, bg, room_cells, directions, movers)
    actual = main(grid)
    return bool(np.array_equal(actual, expected) and not np.array_equal(actual, grid))


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)

    palette = [color for color in range(10) if color != MOVER_COLOR]
    bg = random.choice(palette)
    palette.remove(bg)
    wall = random.choice(palette)
    palette.remove(wall)
    legend_fill, legend_center, legend_indicator = random.sample(palette, 3)

    grid = np.full((height, width), bg, dtype=int)
    grid[0, :] = wall
    grid[-1, :] = wall
    grid[:, 0] = wall
    grid[:, -1] = wall

    rooms: dict[int, tuple[int, int, int, int]] = {0: (1, 1, height - 2, width - 2)}
    next_room_id = 1
    target_rooms = random.randint(*ROOM_COUNT_RANGE)

    while len(rooms) < target_rooms:
        options: list[tuple[int, str, list[int]]] = []
        for room_id, (r0, c0, r1, c1) in rooms.items():
            if r1 - r0 + 1 >= 2 * MIN_ROOM_SPAN + 1:
                choices = list(range(r0 + MIN_ROOM_SPAN, r1 - MIN_ROOM_SPAN + 1))
                if choices:
                    options.append((room_id, "h", choices))
            if c1 - c0 + 1 >= 2 * MIN_ROOM_SPAN + 1:
                choices = list(range(c0 + MIN_ROOM_SPAN, c1 - MIN_ROOM_SPAN + 1))
                if choices:
                    options.append((room_id, "v", choices))
        if not options:
            break

        room_id, orient, choices = random.choice(options)
        r0, c0, r1, c1 = rooms.pop(room_id)
        split = random.choice(choices)
        if orient == "h":
            grid[split, c0 : c1 + 1] = wall
            rooms[next_room_id] = (r0, c0, split - 1, c1)
            rooms[next_room_id + 1] = (split + 1, c0, r1, c1)
        else:
            grid[r0 : r1 + 1, split] = wall
            rooms[next_room_id] = (r0, c0, r1, split - 1)
            rooms[next_room_id + 1] = (r0, split + 1, r1, c1)
        next_room_id += 2

    if len(rooms) < ROOM_COUNT_RANGE[0]:
        return None

    graph = _room_graph(rooms)
    edge_count = sum(len(neighbors) for neighbors in graph.values()) // 2
    if edge_count != len(rooms) - 1:
        return None

    site_candidates: list[tuple[int, tuple[str, int], dict[int, int]]] = []
    for room_id, rect in rooms.items():
        distances = _bfs_distances(graph, room_id)
        if len(distances) != len(rooms) or max(distances.values()) < 2:
            continue
        for site in _legend_sites(rect, height, width):
            site_candidates.append((room_id, site, distances))
    if not site_candidates:
        return None

    anchor_room, site, distances = random.choice(site_candidates)
    anchor_dir = random.choice(CARDINAL_DIRS)
    legend_cells = _paint_legend(
        grid,
        rooms[anchor_room],
        site,
        legend_fill,
        legend_center,
        legend_indicator,
        anchor_dir,
    )

    room_cells = _room_cells_from_layout(rooms, anchor_room, legend_cells)
    if not all(room_cells.values()):
        return None
    if not all(_cells_connected(cells) for cells in room_cells.values()):
        return None

    directions = {
        room_id: anchor_dir if distances[room_id] % 2 == 0 else _opposite(anchor_dir)
        for room_id in rooms
    }
    movers = _choose_movers(room_cells, directions)
    if movers is None:
        return None

    for room_id, points in movers.items():
        for r, c in points:
            grid[r, c] = MOVER_COLOR

    if _get_bg(grid) != bg:
        return None
    if not _validate_generated(grid, bg, room_cells, graph, anchor_room, anchor_dir, directions, movers):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid unique-legend alternating-wall sample")
