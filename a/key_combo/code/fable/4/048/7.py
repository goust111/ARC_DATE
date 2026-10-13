import random
from collections import Counter, deque

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
SHAPES = (
    np.array([[1, 1], [1, 1]], dtype=int),
    np.array([[1, 1], [1, 0]], dtype=int),
    np.array([[1, 1], [0, 1]], dtype=int),
    np.array([[1, 0], [1, 1]], dtype=int),
    np.array([[0, 1], [1, 1]], dtype=int),
)
COARSE_STEP = 4
GEN_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _split_input(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 3
    separator = None
    for r in range(1, g.shape[0]):
        if np.all(g[r] == BLACK):
            separator = r
            break
    assert separator is not None and separator > 0 and separator + 1 < g.shape[0]
    legend = g[:separator, :]
    main = g[separator + 1 :, :]
    assert legend.size > 0 and main.size > 0
    return legend, main


def _legend_cycle(legend: np.ndarray) -> list[int]:
    cycle = [int(v) for v in legend.ravel() if int(v) != BLACK]
    assert len(cycle) >= 2
    return cycle


def _find_gray_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    seen = np.zeros((H, W), dtype=bool)
    objects: list[dict] = []
    for r in range(H):
        for c in range(W):
            if g[r, c] != GREY or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and g[nr, nc] == GREY:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            arr = np.array(coords, dtype=int)
            objects.append({"coords": arr})
    assert objects
    return objects


def _outline_mask(shape: tuple[int, int], coords: np.ndarray) -> np.ndarray:
    H, W = shape
    obj = np.zeros((H, W), dtype=bool)
    obj[coords[:, 0], coords[:, 1]] = True
    out = np.zeros((H, W), dtype=bool)
    for r, c in coords:
        for nr in range(int(r) - 1, int(r) + 2):
            for nc in range(int(c) - 1, int(c) + 2):
                if 0 <= nr < H and 0 <= nc < W and not obj[nr, nc]:
                    out[nr, nc] = True
    return out


def _masks_touch(mask_a: np.ndarray, mask_b: np.ndarray) -> bool:
    coords_a = np.argwhere(mask_a)
    coords_b = np.argwhere(mask_b)
    if coords_a.size == 0 or coords_b.size == 0:
        return False
    diff = np.abs(coords_a[:, None, :] - coords_b[None, :, :])
    return bool(np.any(np.max(diff, axis=2) <= 1))


def _build_object_data(main: np.ndarray) -> list[dict]:
    objects = _find_gray_objects(main)
    for obj in objects:
        obj["outline"] = _outline_mask(main.shape, obj["coords"])
    return objects


def _find_source(objects: list[dict], main: np.ndarray, cycle: list[int]) -> tuple[int, int]:
    source_candidates = []
    cycle_set = set(cycle)
    for idx, obj in enumerate(objects):
        colors = set(map(int, main[obj["outline"]].ravel())) - {BLACK, GREY}
        if not colors:
            continue
        assert len(colors) == 1
        color = int(next(iter(colors)))
        assert color in cycle_set
        source_candidates.append((idx, color))
    assert len(source_candidates) == 1
    return source_candidates[0]


def _object_graph(objects: list[dict]) -> list[list[int]]:
    n = len(objects)
    graph = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if _masks_touch(objects[i]["outline"], objects[j]["outline"]):
                graph[i].append(j)
                graph[j].append(i)
    return graph


def _bfs_distances(graph: list[list[int]], source: int) -> list[int]:
    dist = [-1] * len(graph)
    dist[source] = 0
    queue = deque([source])
    while queue:
        cur = queue.popleft()
        for nxt in graph[cur]:
            if dist[nxt] != -1:
                continue
            dist[nxt] = dist[cur] + 1
            queue.append(nxt)
    assert all(d >= 0 for d in dist)
    return dist


def _render_output(main: np.ndarray, objects: list[dict], cycle: list[int], source_idx: int, source_color: int) -> np.ndarray:
    source_start = cycle.index(source_color)
    dist = _bfs_distances(_object_graph(objects), source_idx)
    out = np.where(main == GREY, GREY, BLACK)
    for idx, obj in enumerate(objects):
        color = cycle[(source_start + dist[idx]) % len(cycle)]
        out[obj["outline"]] = color
    out[main == GREY] = GREY
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Outline each gray object by cycling the legend colors according to graph distance from the source object.

    Input structure:
    - The top of the grid contains a color legend band, followed by a full black separator row.
    - Below the separator lies the main black canvas with several disconnected gray objects.
    - Exactly one gray object is already surrounded by a one-cell-thick colored 8-neighborhood outline, and that color appears in the legend cycle.

    Rule:
    - Remove the legend and separator from the output, keeping only the main canvas.
    - Treat two gray objects as adjacent when their one-cell-thick 8-neighborhood outline masks touch or overlap.
    - Build the object adjacency graph, compute each object's shortest-path hop count from the uniquely pre-outlined source object, and outline the object with the legend color reached by advancing that many steps around the cycle from the source color.

    Key insight:
    - The correct outline color is determined by hop count in the object graph, not by straight-line proximity on the canvas; nearby objects can still be many graph steps away.
    """
    legend, main_grid = _split_input(np.asarray(grid, dtype=int))
    cycle = _legend_cycle(legend)
    objects = _build_object_data(main_grid)
    source_idx, source_color = _find_source(objects, main_grid, cycle)
    return _render_output(main_grid, objects, cycle, source_idx, source_color)


def _grow_positions(rows: int, cols: int, count: int) -> list[tuple[int, int]] | None:
    start = (random.randrange(rows), random.randrange(cols))
    chosen = [start]
    chosen_set = {start}
    while len(chosen) < count:
        frontier = []
        for r, c in chosen:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in chosen_set:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        pos = random.choice(frontier)
        chosen.append(pos)
        chosen_set.add(pos)
    return chosen


def _coarse_graph(positions: list[tuple[int, int]]) -> list[list[int]]:
    n = len(positions)
    graph = [[] for _ in range(n)]
    for i, (r1, c1) in enumerate(positions):
        for j in range(i + 1, n):
            r2, c2 = positions[j]
            if max(abs(r1 - r2), abs(c1 - c2)) <= 1:
                graph[i].append(j)
                graph[j].append(i)
    return graph


def _physical_distances(positions: list[tuple[int, int]], source: int) -> list[float]:
    sr, sc = positions[source]
    out = []
    for r, c in positions:
        dr = (r - sr) * COARSE_STEP
        dc = (c - sc) * COARSE_STEP
        out.append(float(np.hypot(dr, dc)))
    return out


def _pick_source(positions: list[tuple[int, int]], graph: list[list[int]]) -> int | None:
    for source in random.sample(list(range(len(positions))), k=len(positions)):
        dist = _bfs_distances(graph, source)
        if max(dist) < 3:
            continue
        phys = _physical_distances(positions, source)
        good = False
        for i in range(len(positions)):
            for j in range(len(positions)):
                if i == j:
                    continue
                if dist[i] > dist[j] and phys[i] + 1e-6 < phys[j]:
                    good = True
                    break
            if good:
                break
        if good:
            return source
    return None


def _render_main(
    positions: list[tuple[int, int]],
    shapes: list[np.ndarray],
    margins: tuple[int, int, int, int],
) -> tuple[np.ndarray, list[dict]]:
    top, bottom, left, right = margins
    min_r = min(r for r, _ in positions)
    max_r = max(r for r, _ in positions)
    min_c = min(c for _, c in positions)
    max_c = max(c for _, c in positions)
    H = top + (max_r - min_r) * COARSE_STEP + 2 + bottom
    W = left + (max_c - min_c) * COARSE_STEP + 2 + right
    main_grid = np.full((H, W), BLACK, dtype=int)
    objects = []
    for (cr, cc), shape in zip(positions, shapes):
        r0 = top + (cr - min_r) * COARSE_STEP
        c0 = left + (cc - min_c) * COARSE_STEP
        coords = []
        for dr in range(2):
            for dc in range(2):
                if int(shape[dr, dc]) == 0:
                    continue
                rr = r0 + dr
                cc2 = c0 + dc
                main_grid[rr, cc2] = GREY
                coords.append((rr, cc2))
        objects.append({"coords": np.array(coords, dtype=int)})
    for obj in objects:
        obj["outline"] = _outline_mask(main_grid.shape, obj["coords"])
    return main_grid, objects


def _build_input_from_plan(
    positions: list[tuple[int, int]],
    shapes: list[np.ndarray],
    cycle: list[int],
    source_idx: int,
    source_color: int,
) -> tuple[np.ndarray, np.ndarray]:
    margins = (
        random.randint(1, 3),
        random.randint(1, 3),
        random.randint(1, 3),
        random.randint(1, 3),
    )
    main_grid, objects = _render_main(positions, shapes, margins)
    expected = _render_output(main_grid, objects, cycle, source_idx, source_color)

    inp_main = main_grid.copy()
    inp_main[objects[source_idx]["outline"]] = source_color

    total_h = inp_main.shape[0] + 2
    total_w = inp_main.shape[1]
    if total_h > 30 or total_w > 30:
        raise ValueError("grid exceeds ARC size limit")

    inp = np.full((total_h, total_w), BLACK, dtype=int)
    start = random.randint(0, total_w - len(cycle))
    inp[0, start : start + len(cycle)] = np.array(cycle, dtype=int)
    inp[2:, :] = inp_main
    return inp, expected


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(4, 6)
    cols = random.randint(4, 6)
    count = random.randint(5, 8)
    positions = _grow_positions(rows, cols, count)
    if positions is None:
        return None

    cycle_len = random.randint(3, 5)
    cycle = random.sample(PALETTE, cycle_len)
    shapes = [random.choice(SHAPES) for _ in positions]

    probe_main, probe_objects = _render_main(positions, shapes, (1, 1, 1, 1))
    actual_graph = _object_graph(probe_objects)
    if any(not nbrs for nbrs in actual_graph):
        return None
    source_idx = _pick_source(positions, actual_graph)
    if source_idx is None:
        return None
    source_color = random.choice(cycle)

    try:
        inp, expected = _build_input_from_plan(positions, shapes, cycle, source_idx, source_color)
    except ValueError:
        return None

    try:
        got = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(got, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
