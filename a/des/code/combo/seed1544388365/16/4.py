from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
GREY = 5
SOURCE_PALETTE = np.array([2, 3, 4, 6, 7, 8, 9], dtype=int)
GENERATION_ATTEMPTS = 300
MODES = ("single", "multi", "conflict")


def _find_separators(grid: np.ndarray) -> tuple[list[int], list[int]]:
    h, w = grid.shape
    sep_rows = [
        r
        for r in range(h)
        if int(np.count_nonzero(grid[r] == BLUE)) > w // 2
        and bool(np.isin(grid[r], [BLACK, BLUE]).all())
    ]
    sep_cols = [
        c
        for c in range(w)
        if int(np.count_nonzero(grid[:, c] == BLUE)) > h // 2
        and bool(np.isin(grid[:, c], [BLACK, BLUE]).all())
    ]
    return sep_rows, sep_cols


def _room_color(patch: np.ndarray) -> int | None:
    vals = np.unique(patch)
    if vals.size == 1:
        value = int(vals[0])
        if value == BLACK:
            return None
        if value != BLUE:
            return value
        return None

    colors = [int(v) for v in vals if int(v) not in (BLACK, BLUE)]
    if len(colors) == 1 and bool(np.isin(patch, [BLACK, colors[0]]).all()):
        return colors[0]
    return None


def _room_structure(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    sep_rows, sep_cols = _find_separators(g)

    row_bounds = [-1, *sep_rows, h]
    col_bounds = [-1, *sep_cols, w]
    rooms = []
    room_ids = {}

    for ri in range(len(row_bounds) - 1):
        r0 = row_bounds[ri] + 1
        r1 = row_bounds[ri + 1] - 1
        for ci in range(len(col_bounds) - 1):
            c0 = col_bounds[ci] + 1
            c1 = col_bounds[ci + 1] - 1
            room_ids[(ri, ci)] = len(rooms)
            patch = g[r0 : r1 + 1, c0 : c1 + 1]
            rooms.append(
                {
                    "band": (ri, ci),
                    "bbox": (r0, r1, c0, c1),
                    "color": _room_color(patch),
                }
            )

    graph = [set() for _ in rooms]
    n_row_bands = len(row_bounds) - 1
    n_col_bands = len(col_bounds) - 1

    for ri in range(n_row_bands):
        r0 = row_bounds[ri] + 1
        r1 = row_bounds[ri + 1] - 1
        for ci in range(n_col_bands - 1):
            sep_c = col_bounds[ci + 1]
            if bool(np.any(g[r0 : r1 + 1, sep_c] == BLACK)):
                a = room_ids[(ri, ci)]
                b = room_ids[(ri, ci + 1)]
                graph[a].add(b)
                graph[b].add(a)

    for ri in range(n_row_bands - 1):
        sep_r = row_bounds[ri + 1]
        for ci in range(n_col_bands):
            c0 = col_bounds[ci] + 1
            c1 = col_bounds[ci + 1] - 1
            if bool(np.any(g[sep_r, c0 : c1 + 1] == BLACK)):
                a = room_ids[(ri, ci)]
                b = room_ids[(ri + 1, ci)]
                graph[a].add(b)
                graph[b].add(a)

    return rooms, graph, sep_rows, sep_cols


def _bfs_distances(graph: list[set[int]], start: int) -> list[int]:
    dist = [-1] * len(graph)
    dist[start] = 0
    q = deque([start])
    while q:
        cur = q.popleft()
        for nxt in graph[cur]:
            if dist[nxt] != -1:
                continue
            dist[nxt] = dist[cur] + 1
            q.append(nxt)
    return dist


def _final_room_colors(rooms: list[dict], graph: list[set[int]]) -> list[int | None]:
    sources = [(idx, int(room["color"])) for idx, room in enumerate(rooms) if room["color"] is not None]
    if not sources:
        return [None] * len(rooms)

    source_dists = [(color, _bfs_distances(graph, idx)) for idx, color in sources]
    labels: list[int | None] = []
    for idx, room in enumerate(rooms):
        if room["color"] is not None:
            labels.append(int(room["color"]))
            continue

        best_dist = None
        best_colors: set[int] = set()
        for color, dist in source_dists:
            if dist[idx] < 0:
                continue
            if best_dist is None or dist[idx] < best_dist:
                best_dist = dist[idx]
                best_colors = {int(color)}
            elif dist[idx] == best_dist:
                best_colors.add(int(color))

        if best_dist is None:
            labels.append(None)
        elif len(best_colors) == 1:
            labels.append(next(iter(best_colors)))
        else:
            labels.append(GREY)
    return labels


def _render_solution(grid: np.ndarray, rooms: list[dict], labels: list[int | None]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for room, color in zip(rooms, labels):
        if color is None:
            continue
        r0, r1, c0, c1 = room["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify blue separator lines, interpret black gaps on them as doors, and
    propagate room colors through the door graph. Empty rooms take the nearest
    source color; equal-distance conflicts become grey.
    """
    g = np.asarray(grid, dtype=int)
    rooms, graph, _sep_rows, _sep_cols = _room_structure(g)
    labels = _final_room_colors(rooms, graph)
    return _render_solution(g, rooms, labels)


def _layout_from_sizes(heights: list[int], widths: list[int]):
    row_starts = []
    sep_rows = []
    cursor = 0
    for i, height in enumerate(heights):
        row_starts.append(cursor)
        cursor += int(height)
        if i + 1 < len(heights):
            sep_rows.append(cursor)
            cursor += 1
    h = cursor

    col_starts = []
    sep_cols = []
    cursor = 0
    for i, width in enumerate(widths):
        col_starts.append(cursor)
        cursor += int(width)
        if i + 1 < len(widths):
            sep_cols.append(cursor)
            cursor += 1
    w = cursor

    boxes = []
    for ri, r0 in enumerate(row_starts):
        for ci, c0 in enumerate(col_starts):
            boxes.append((r0, r0 + heights[ri] - 1, c0, c0 + widths[ci] - 1))
    return (h, w), row_starts, col_starts, sep_rows, sep_cols, boxes


def _edge_specs(
    heights: list[int],
    widths: list[int],
    row_starts: list[int],
    col_starts: list[int],
    sep_rows: list[int],
    sep_cols: list[int],
):
    nrows = len(heights)
    ncols = len(widths)
    specs = []
    pair_to_spec = {}
    neighbors = [[] for _ in range(nrows * ncols)]

    for ri in range(nrows):
        for ci in range(ncols - 1):
            a = ri * ncols + ci
            b = a + 1
            spec = {
                "a": a,
                "b": b,
                "kind": "v",
                "sep": sep_cols[ci],
                "r0": row_starts[ri],
                "r1": row_starts[ri] + heights[ri] - 1,
            }
            specs.append(spec)
            key = (a, b)
            pair_to_spec[key] = spec
            neighbors[a].append(b)
            neighbors[b].append(a)

    for ri in range(nrows - 1):
        for ci in range(ncols):
            a = ri * ncols + ci
            b = (ri + 1) * ncols + ci
            spec = {
                "a": a,
                "b": b,
                "kind": "h",
                "sep": sep_rows[ri],
                "c0": col_starts[ci],
                "c1": col_starts[ci] + widths[ci] - 1,
            }
            specs.append(spec)
            key = (a, b)
            pair_to_spec[key] = spec
            neighbors[a].append(b)
            neighbors[b].append(a)

    return specs, pair_to_spec, neighbors


def _door_graph(n_rooms: int, door_pairs: set[tuple[int, int]]) -> list[set[int]]:
    graph = [set() for _ in range(n_rooms)]
    for a, b in door_pairs:
        graph[a].add(b)
        graph[b].add(a)
    return graph


def _components(graph: list[set[int]]) -> list[list[int]]:
    seen = set()
    comps = []
    for start in range(len(graph)):
        if start in seen:
            continue
        q = deque([start])
        seen.add(start)
        comp = []
        while q:
            cur = q.popleft()
            comp.append(cur)
            for nxt in graph[cur]:
                if nxt in seen:
                    continue
                seen.add(nxt)
                q.append(nxt)
        comps.append(comp)
    return comps


def _grow_component(
    rng: np.random.Generator,
    seed: int,
    target_size: int,
    available: set[int],
    neighbors: list[list[int]],
):
    comp = {seed}
    door_pairs: set[tuple[int, int]] = set()

    while len(comp) < target_size:
        frontier = []
        for room in comp:
            for nxt in neighbors[room]:
                if nxt in available and nxt not in comp:
                    frontier.append((room, nxt))
        if not frontier:
            return None
        room, nxt = frontier[int(rng.integers(0, len(frontier)))]
        comp.add(int(nxt))
        door_pairs.add(tuple(sorted((int(room), int(nxt)))))

    return comp, door_pairs


def _single_mode_doors(
    rng: np.random.Generator,
    n_rooms: int,
    neighbors: list[list[int]],
    pair_to_spec: dict[tuple[int, int], dict],
):
    available = set(range(n_rooms))
    seed = int(rng.integers(0, n_rooms))
    target_size = int(rng.integers(3, n_rooms + 1))
    grown = _grow_component(rng, seed, target_size, available, neighbors)
    if grown is None:
        return None
    comp, door_pairs = grown

    for a in comp:
        for b in neighbors[a]:
            key = tuple(sorted((int(a), int(b))))
            if b in comp and key not in door_pairs and rng.random() < 0.35:
                door_pairs.add(key)

    leftovers = list(set(range(n_rooms)) - comp)
    for room in leftovers:
        for nxt in neighbors[room]:
            key = tuple(sorted((int(room), int(nxt))))
            if nxt in leftovers and key in pair_to_spec and rng.random() < 0.12:
                door_pairs.add(key)

    source_rooms = [seed]
    return door_pairs, source_rooms


def _multi_mode_doors(
    rng: np.random.Generator,
    n_rooms: int,
    neighbors: list[list[int]],
    pair_to_spec: dict[tuple[int, int], dict],
):
    max_components = min(3, n_rooms // 2)
    if max_components < 2:
        return None
    component_count = int(rng.integers(2, max_components + 1))
    available = set(range(n_rooms))
    chosen_components = []
    door_pairs: set[tuple[int, int]] = set()

    for comp_idx in range(component_count):
        if len(available) < 2 * (component_count - comp_idx):
            return None
        seed = int(rng.choice(list(available)))
        max_size = min(5, len(available) - 2 * (component_count - comp_idx - 1))
        if max_size < 2:
            return None
        target_size = int(rng.integers(2, max_size + 1))
        grown = _grow_component(rng, seed, target_size, available, neighbors)
        if grown is None:
            return None
        comp, comp_pairs = grown
        chosen_components.append(list(comp))
        door_pairs |= comp_pairs
        available -= comp

    for comp in chosen_components:
        comp_set = set(comp)
        for a in comp:
            for b in neighbors[a]:
                key = tuple(sorted((int(a), int(b))))
                if b in comp_set and key not in door_pairs and rng.random() < 0.25:
                    door_pairs.add(key)

    leftovers = list(available)
    for room in leftovers:
        for nxt in neighbors[room]:
            key = tuple(sorted((int(room), int(nxt))))
            if nxt in available and key in pair_to_spec and rng.random() < 0.10:
                door_pairs.add(key)

    source_rooms = [int(rng.choice(comp)) for comp in chosen_components]
    return door_pairs, source_rooms


def _conflict_mode_doors(
    rng: np.random.Generator,
    nrows: int,
    ncols: int,
    specs: list[dict],
):
    door_pairs = {tuple(sorted((int(spec["a"]), int(spec["b"])))) for spec in specs}
    return door_pairs


def _conflict_source_rooms(
    rng: np.random.Generator,
    nrows: int,
    ncols: int,
    graph: list[set[int]],
):
    candidates = []

    if ncols >= 2:
        for ri in range(nrows):
            candidates.append((ri * ncols, ri * ncols + (ncols - 1)))
    if nrows >= 2:
        for ci in range(ncols):
            candidates.append((ci, (nrows - 1) * ncols + ci))
    candidates.append((0, nrows * ncols - 1))
    candidates.append((ncols - 1, (nrows - 1) * ncols))

    unique_candidates = []
    seen = set()
    for a, b in candidates:
        key = tuple(sorted((int(a), int(b))))
        if a == b or key in seen:
            continue
        seen.add(key)
        unique_candidates.append((int(a), int(b)))

    rng.shuffle(unique_candidates)
    for a, b in unique_candidates:
        rooms = [{"color": None} for _ in range(nrows * ncols)]
        rooms[a]["color"] = 2
        rooms[b]["color"] = 3
        labels = _final_room_colors(rooms, graph)
        if GREY in [labels[idx] for idx, room in enumerate(rooms) if room["color"] is None]:
            return [a, b]
    return None


def _paint_doors(
    grid: np.ndarray,
    rng: np.random.Generator,
    pair_to_spec: dict[tuple[int, int], dict],
    door_pairs: set[tuple[int, int]],
):
    for pair in door_pairs:
        spec = pair_to_spec[pair]
        if spec["kind"] == "v":
            row = int(rng.integers(spec["r0"], spec["r1"] + 1))
            grid[row, spec["sep"]] = BLACK
        else:
            col = int(rng.integers(spec["c0"], spec["c1"] + 1))
            grid[spec["sep"], col] = BLACK


def _mode_ok(mode: str, rooms: list[dict], labels: list[int | None]) -> bool:
    newly_filled = [
        labels[idx]
        for idx, room in enumerate(rooms)
        if room["color"] is None and labels[idx] is not None
    ]
    if not newly_filled:
        return False

    if mode == "single":
        return len(newly_filled) >= 2 and GREY not in newly_filled and len(set(newly_filled)) == 1

    if mode == "multi":
        non_grey = {int(color) for color in newly_filled if color != GREY}
        return GREY not in newly_filled and len(non_grey) >= 2

    if mode == "conflict":
        return GREY in newly_filled

    return False


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    mode = MODES[int(rng.integers(0, len(MODES)))]
    nrows = int(rng.integers(2, 6))
    ncols = int(rng.integers(2, 6))
    heights = [int(v) for v in rng.integers(2, 6, size=nrows)]
    widths = [int(v) for v in rng.integers(2, 6, size=ncols)]

    (h, w), row_starts, col_starts, sep_rows, sep_cols, boxes = _layout_from_sizes(heights, widths)
    grid = np.full((h, w), BLACK, dtype=int)
    for row in sep_rows:
        grid[row, :] = BLUE
    for col in sep_cols:
        grid[:, col] = BLUE

    specs, pair_to_spec, neighbors = _edge_specs(heights, widths, row_starts, col_starts, sep_rows, sep_cols)
    n_rooms = nrows * ncols

    if mode == "single":
        chosen = _single_mode_doors(rng, n_rooms, neighbors, pair_to_spec)
    elif mode == "multi":
        chosen = _multi_mode_doors(rng, n_rooms, neighbors, pair_to_spec)
    else:
        door_pairs = _conflict_mode_doors(rng, nrows, ncols, specs)
        source_rooms = _conflict_source_rooms(rng, nrows, ncols, _door_graph(n_rooms, door_pairs))
        chosen = None if source_rooms is None else (door_pairs, source_rooms)
    if chosen is None:
        return None

    door_pairs, source_rooms = chosen
    _paint_doors(grid, rng, pair_to_spec, door_pairs)

    color_count = len(source_rooms)
    source_colors = rng.choice(SOURCE_PALETTE, size=color_count, replace=False)
    for room_idx, color in zip(source_rooms, source_colors):
        r0, r1, c0, c1 = boxes[int(room_idx)]
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)

    rooms, graph, found_sep_rows, found_sep_cols = _room_structure(grid)
    if len(rooms) != n_rooms:
        return None
    if found_sep_rows != sep_rows or found_sep_cols != sep_cols:
        return None

    parsed_pairs = {
        tuple(sorted((a, b)))
        for a, neigh in enumerate(graph)
        for b in neigh
        if a < b
    }
    if parsed_pairs != door_pairs:
        return None

    if sum(room["color"] is not None for room in rooms) != len(source_rooms):
        return None

    labels = _final_room_colors(rooms, graph)
    if not _mode_ok(mode, rooms, labels):
        return None

    out = _render_solution(grid, rooms, labels)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed to create a valid room-diffusion puzzle")
