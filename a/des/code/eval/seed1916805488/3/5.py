import random
from collections import defaultdict, deque

import numpy as np


BLACK = 0
GREY = 5
COLOR_CYCLE = (2, 1, 3, 4)  # red -> blue -> green -> yellow


def _next_cycle_color(color: int, steps: int) -> int:
    idx = COLOR_CYCLE.index(int(color))
    return int(COLOR_CYCLE[(idx + steps) % len(COLOR_CYCLE)])


def _wall_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    non_black = [(int(v), int(c)) for v, c in zip(values, counts) if int(v) != BLACK]
    assert non_black, "grid must contain a wall color"
    return max(non_black, key=lambda item: item[1])[0]


def _wall_lines(grid: np.ndarray, wall: int) -> tuple[list[int], list[int]]:
    h, w = grid.shape
    wall_rows = [r for r in range(h) if int(np.count_nonzero(grid[r] == wall)) > w // 2]
    wall_cols = [c for c in range(w) if int(np.count_nonzero(grid[:, c] == wall)) > h // 2]
    assert len(wall_rows) >= 2 and len(wall_cols) >= 2
    assert wall_rows[0] == 0 and wall_rows[-1] == h - 1
    assert wall_cols[0] == 0 and wall_cols[-1] == w - 1
    return wall_rows, wall_cols


def _room_intervals(lines: list[int]) -> list[tuple[int, int]]:
    spans = []
    for a, b in zip(lines, lines[1:]):
        if b - a > 1:
            spans.append((a + 1, b - 1))
    assert spans, "expected at least one room span"
    return spans


def _room_layout(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]], dict[tuple[int, int], tuple[int, int, int, int]]]:
    wall = _wall_color(grid)
    wall_rows, wall_cols = _wall_lines(grid, wall)
    row_spans = _room_intervals(wall_rows)
    col_spans = _room_intervals(wall_cols)
    rooms: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for ri, (r0, r1) in enumerate(row_spans):
        for ci, (c0, c1) in enumerate(col_spans):
            rooms[(ri, ci)] = (r0, r1, c0, c1)
    return row_spans, col_spans, rooms


def _room_graph(grid: np.ndarray, wall: int, row_spans: list[tuple[int, int]], col_spans: list[tuple[int, int]]) -> dict[tuple[int, int], list[tuple[int, int]]]:
    graph: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    room_rows = len(row_spans)
    room_cols = len(col_spans)

    for ri in range(room_rows):
        for ci in range(room_cols):
            graph[(ri, ci)]

    for boundary_idx in range(room_rows - 1):
        wr = row_spans[boundary_idx][1] + 1
        for ci, (c0, c1) in enumerate(col_spans):
            segment = grid[wr, c0 : c1 + 1]
            if np.any(segment != wall):
                a = (boundary_idx, ci)
                b = (boundary_idx + 1, ci)
                graph[a].append(b)
                graph[b].append(a)

    for boundary_idx in range(room_cols - 1):
        wc = col_spans[boundary_idx][1] + 1
        for ri, (r0, r1) in enumerate(row_spans):
            segment = grid[r0 : r1 + 1, wc]
            if np.any(segment != wall):
                a = (ri, boundary_idx)
                b = (ri, boundary_idx + 1)
                graph[a].append(b)
                graph[b].append(a)

    return graph


def _seed_info(grid: np.ndarray, wall: int, rooms: dict[tuple[int, int], tuple[int, int, int, int]]) -> dict[tuple[int, int], tuple[int, int]]:
    sources: dict[tuple[int, int], tuple[int, int]] = {}
    for room_id, (r0, r1, c0, c1) in rooms.items():
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        seeds = patch[(patch != BLACK) & (patch != wall)]
        if len(seeds) == 0:
            continue
        colors = set(map(int, seeds.tolist()))
        assert len(colors) == 1
        color = next(iter(colors))
        assert color in COLOR_CYCLE
        assert 1 <= len(seeds) <= 3
        sources[room_id] = (color, int(len(seeds)))
    assert sources, "expected at least one source room"
    return sources


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    wall = _wall_color(g)
    row_spans, col_spans, rooms = _room_layout(g)
    graph = _room_graph(g, wall, row_spans, col_spans)
    sources = _seed_info(g, wall, rooms)

    arrivals: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)

    for source_room, (base_color, strength) in sources.items():
        dist = {source_room: 0}
        q = deque([source_room])
        while q:
            room = q.popleft()
            if dist[room] == strength:
                continue
            for nxt in graph[room]:
                if nxt not in dist:
                    dist[nxt] = dist[room] + 1
                    q.append(nxt)
        for room, d in dist.items():
            if 1 <= d <= strength:
                arrivals[room].append((d, _next_cycle_color(base_color, d - 1)))

    out = g.copy()
    for room, hits in arrivals.items():
        best_dist = min(d for d, _ in hits)
        best_hits = [(d, color) for d, color in hits if d == best_dist]
        if len(best_hits) != 1:
            continue
        fill_color = best_hits[0][1]
        r0, r1, c0, c1 = rooms[room]
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        patch[patch == BLACK] = fill_color

    return out


def _choose_spans(rng: random.Random, count: int) -> list[int]:
    return [rng.randint(3, 6) for _ in range(count)]


def _positions_from_sizes(sizes: list[int]) -> list[tuple[int, int]]:
    spans = []
    start = 1
    for size in sizes:
        spans.append((start, start + size - 1))
        start += size + 1
    return spans


def _build_room_graph(room_rows: int, room_cols: int) -> dict[tuple[int, int], list[tuple[int, int]]]:
    graph: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for ri in range(room_rows):
        for ci in range(room_cols):
            if ri + 1 < room_rows:
                graph[(ri, ci)].append((ri + 1, ci))
                graph[(ri + 1, ci)].append((ri, ci))
            if ci + 1 < room_cols:
                graph[(ri, ci)].append((ri, ci + 1))
                graph[(ri, ci + 1)].append((ri, ci))
    return graph


def _distances(graph: dict[tuple[int, int], list[tuple[int, int]]], start: tuple[int, int]) -> dict[tuple[int, int], int]:
    dist = {start: 0}
    q = deque([start])
    while q:
        node = q.popleft()
        for nxt in graph[node]:
            if nxt not in dist:
                dist[nxt] = dist[node] + 1
                q.append(nxt)
    return dist


def _draw_walls(grid: np.ndarray, row_spans: list[tuple[int, int]], col_spans: list[tuple[int, int]]) -> None:
    for r0, r1 in row_spans:
        grid[r0 - 1, :] = GREY
        grid[r1 + 1, :] = GREY
    for c0, c1 in col_spans:
        grid[:, c0 - 1] = GREY
        grid[:, c1 + 1] = GREY


def _add_doors(grid: np.ndarray, rng: random.Random, row_spans: list[tuple[int, int]], col_spans: list[tuple[int, int]]) -> None:
    room_rows = len(row_spans)
    room_cols = len(col_spans)

    for ri in range(room_rows - 1):
        wall_r = row_spans[ri][1] + 1
        for ci, (c0, c1) in enumerate(col_spans):
            door_c = rng.randint(c0, c1)
            grid[wall_r, door_c] = BLACK

    for ci in range(room_cols - 1):
        wall_c = col_spans[ci][1] + 1
        for ri, (r0, r1) in enumerate(row_spans):
            door_r = rng.randint(r0, r1)
            grid[door_r, wall_c] = BLACK


def _place_seeds(
    grid: np.ndarray,
    rng: random.Random,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
) -> None:
    room_rows = len(row_spans)
    room_cols = len(col_spans)
    rooms = [(ri, ci) for ri in range(room_rows) for ci in range(room_cols)]
    graph = _build_room_graph(room_rows, room_cols)

    focus_strength = rng.choice([1, 2, 3])
    candidate_focus_rooms = [room for room in rooms if max(_distances(graph, room).values()) >= focus_strength]
    if not candidate_focus_rooms:
        raise RuntimeError("no room can support the requested propagation depth")

    source_count = rng.randint(1, min(4, len(rooms)))
    chosen = [rng.choice(candidate_focus_rooms)]
    remaining = [room for room in rooms if room != chosen[0]]
    rng.shuffle(remaining)
    chosen.extend(remaining[: source_count - 1])

    available_colors = list(COLOR_CYCLE)
    rng.shuffle(available_colors)
    source_colors = available_colors[:source_count]

    strengths = [focus_strength]
    for room in chosen[1:]:
        max_depth = min(3, max(_distances(graph, room).values()))
        strengths.append(rng.randint(1, max_depth))

    for room, color, count in zip(chosen, source_colors, strengths):
        r0, r1 = row_spans[room[0]]
        c0, c1 = col_spans[room[1]]
        cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
        rng.shuffle(cells)
        for r, c in cells[:count]:
            grid[r, c] = color


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(300):
        room_rows = rng.randint(2, 4)
        room_cols = rng.randint(2, 4)
        if room_rows + room_cols < 4:
            continue

        row_sizes = _choose_spans(rng, room_rows)
        col_sizes = _choose_spans(rng, room_cols)
        row_spans = _positions_from_sizes(row_sizes)
        col_spans = _positions_from_sizes(col_sizes)

        height = row_spans[-1][1] + 2
        width = col_spans[-1][1] + 2
        grid = np.zeros((height, width), dtype=int)

        _draw_walls(grid, row_spans, col_spans)
        _add_doors(grid, rng, row_spans, col_spans)

        try:
            _place_seeds(grid, rng, row_spans, col_spans)
        except RuntimeError:
            continue

        wall = _wall_color(grid)
        rs, cs, rooms = _room_layout(grid)
        assert rs == row_spans and cs == col_spans
        assert _seed_info(grid, wall, rooms)
        return grid

    raise RuntimeError("generate_input: failed after 300 attempts")
