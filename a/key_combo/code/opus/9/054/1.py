from __future__ import annotations

import hashlib
import random
from collections import deque

import numpy as np


BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

_GEN_ATTEMPTS = 500
_USED_OUTPUT_SIGNATURES: set[str] = set()
_GEN_COUNTER = 0


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _trace_path(shape: tuple[int, int], start: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    r, c = start
    dc = 1
    path: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w:
        path.append((r, c))
        nr = r - 1
        nc = c + dc
        if nr < 0:
            break
        if nc < 0 or nc >= w:
            dc = -dc
            nc = c + dc
        r, c = nr, nc
    return path


def _label_regions(mask: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    regions: list[list[tuple[int, int]]] = []
    next_id = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            next_id += 1
            labels[r, c] = next_id
            q = deque([(r, c)])
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = next_id
                        q.append((nr, nc))
            regions.append(cells)
    return labels, regions


def _build_region_graph(
    labels: np.ndarray, path: list[tuple[int, int]], n_regions: int
) -> dict[int, set[int]]:
    graph = {rid: set() for rid in range(1, n_regions + 1)}
    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        a = (r1, c0)
        b = (r0, c1)
        ra = int(labels[a])
        rb = int(labels[b])
        if ra <= 0 or rb <= 0 or ra == rb:
            raise AssertionError("each diagonal step must separate two distinct regions")
        graph[ra].add(rb)
        graph[rb].add(ra)
    return graph


def _bipartite_color(graph: dict[int, set[int]], start_region: int) -> dict[int, int]:
    colors = {start_region: 0}
    q = deque([start_region])
    while q:
        region = q.popleft()
        for nxt in graph[region]:
            want = 1 - colors[region]
            if nxt in colors:
                if colors[nxt] != want:
                    raise AssertionError("region graph must be bipartite")
                continue
            colors[nxt] = want
            q.append(nxt)
    if set(colors) != set(graph):
        raise AssertionError("all regions must lie in one trajectory-defined adjacency graph")
    return colors


def _ordered_path_regions(graph: dict[int, set[int]]) -> list[int]:
    if len(graph) == 1:
        return [next(iter(graph))]
    ends = sorted([rid for rid, nbrs in graph.items() if len(nbrs) == 1])
    if len(ends) != 2:
        raise AssertionError("trajectory-separated region graph must be a chain")
    ordered = [ends[0]]
    prev = 0
    cur = ends[0]
    while True:
        nxts = [n for n in sorted(graph[cur]) if n != prev]
        if not nxts:
            break
        if len(nxts) != 1:
            raise AssertionError("trajectory-separated region graph must not branch")
        prev, cur = cur, nxts[0]
        ordered.append(cur)
    if len(ordered) != len(graph):
        raise AssertionError("ordered chain traversal missed regions")
    return ordered


def _solve(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("grid must be 2D")

    blue = np.argwhere(g == BLUE)
    red = np.argwhere(g == RED)
    if blue.shape[0] != 1 or red.shape[0] != 1:
        raise AssertionError("expected exactly one blue start and one red seed")

    start = tuple(map(int, blue[0]))
    seed = tuple(map(int, red[0]))
    path = _trace_path(g.shape, start)
    wall = np.zeros(g.shape, dtype=bool)
    for r, c in path:
        wall[r, c] = True
    if wall[seed]:
        raise AssertionError("seed must not lie on the trajectory")

    labels, regions = _label_regions(~wall)
    if len(regions) < 2:
        raise AssertionError("trajectory must split the canvas into multiple regions")
    graph = _build_region_graph(labels, path, len(regions))
    seed_region = int(labels[seed])
    if seed_region <= 0:
        raise AssertionError("seed must lie inside a non-wall region")
    parity = _bipartite_color(graph, seed_region)

    out = np.empty_like(g)
    for rid in range(1, len(regions) + 1):
        out[labels == rid] = GREY if parity[rid] == 0 else YELLOW
    out[wall] = GREEN
    out[start] = BLUE

    meta = {
        "start": start,
        "seed": seed,
        "path": path,
        "wall": wall,
        "labels": labels,
        "regions": regions,
        "graph": graph,
        "ordered": _ordered_path_regions(graph),
        "parity": parity,
        "seed_region": seed_region,
    }
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the bouncing diagonal trajectory as a wall, then alternate-fill the separated regions from the seed side.

    Input structure:
    - The grid is black except for one blue start pixel near the bottom and one red seed pixel away from the trajectory.
    - The blue pixel launches a diagonal ray that always moves upward and reflects only on the left/right borders.
    - The traced trajectory acts as a one-cell-thick separating wall, while all non-trajectory cells form 4-connected regions.

    Rule:
    - Trace the blue pixel up-right, flipping only the horizontal direction whenever the next step would leave the left or right edge.
    - Treat every trajectory cell as a wall; keep the blue start blue and color all other trajectory cells green.
    - Split the remaining cells into 4-connected regions, connect regions that lie on opposite sides of each diagonal trajectory step, and 2-color that region graph.
    - The region containing the red seed is the grey side, every wall-adjacent region across the trajectory alternates to yellow, and the red seed itself is overwritten by its region fill.

    Key insight:
    - The ray is not the answer by itself: its bounce geometry determines the region-adjacency chain, and the red seed chooses which parity class of that chain becomes grey.
    """
    return _solve(grid)[0]


def _select_region_for_seed(
    ordered: list[int],
    regions: list[list[tuple[int, int]]],
    desired_parity: int,
    prefer_end: bool,
) -> int | None:
    candidates = []
    for pos, rid in enumerate(ordered):
        if pos % 2 != desired_parity:
            continue
        size = len(regions[rid - 1])
        if size < 8:
            continue
        edge_distance = min(pos, len(ordered) - 1 - pos)
        candidates.append((edge_distance, -size, rid))
    if not candidates:
        return None
    candidates.sort()
    if not prefer_end:
        candidates = sorted(candidates, key=lambda item: (-item[0], item[1], item[2]))
    top = candidates[: min(3, len(candidates))]
    return int(random.choice(top)[2])


def _pick_seed_cell(
    labels: np.ndarray, wall: np.ndarray, region_id: int, start_col: int
) -> tuple[int, int] | None:
    h, w = labels.shape
    cells = np.argwhere(labels == region_id)
    scored: list[tuple[int, int, int]] = []
    for r0, c0 in cells:
        r, c = int(r0), int(c0)
        if c == start_col:
            continue
        same_neighbors = 0
        wall_neighbors = 0
        border_penalty = 0
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w:
                if labels[nr, nc] == region_id:
                    same_neighbors += 1
                if wall[nr, nc]:
                    wall_neighbors += 1
            else:
                border_penalty += 1
        score = 4 * same_neighbors - 5 * wall_neighbors - 2 * border_penalty
        scored.append((score, r, c))
    if not scored:
        return None
    scored.sort(reverse=True)
    best_score = scored[0][0]
    keep = [cell for cell in scored if cell[0] >= best_score - 2]
    _, r, c = random.choice(keep)
    return r, c


def _choose_dimensions(branch: int) -> tuple[int, int, range]:
    if branch == 0:
        w = random.choice([9, 11, 13, 15])
        h = random.randint(w + 6, min(29, w + 10))
        start_cols = range(3, w - 2)
    elif branch == 1:
        w = random.choice([9, 11, 13, 15])
        h = random.randint(w + 7, min(29, w + 13))
        start_cols = range(2, w - 2)
    elif branch == 2:
        w = random.choice([7, 9])
        h = random.randint(max(18, w + 10), 29)
        start_cols = range(1, w - 1)
    else:
        w = random.choice([11, 13, 15, 17])
        h = random.randint(w + 6, min(29, w + 12))
        start_cols = range(2, w - 2)
    return h, w, start_cols


def _validate_generated(grid: np.ndarray, desired_parity: int) -> bool:
    out, meta = _solve(grid)
    h, w = grid.shape

    if max(h, w) > 30 or w % 2 == 0 or h <= w:
        return False
    if tuple(np.argwhere(grid == BLUE)[0]) != meta["start"]:
        return False
    if tuple(np.argwhere(grid == RED)[0]) != meta["seed"]:
        return False
    if meta["start"][0] != h - 1:
        return False
    if not (0 < meta["start"][1] < w - 1):
        return False
    if meta["seed"][1] == meta["start"][1]:
        return False
    if len(meta["ordered"]) < 4:
        return False
    if meta["parity"][meta["seed_region"]] != 0:
        return False
    if len(set(out.flat)) != 4:
        return False
    if np.count_nonzero(out == GREY) == 0 or np.count_nonzero(out == YELLOW) == 0:
        return False

    graph = meta["graph"]
    for rid, nbrs in graph.items():
        for nbr in nbrs:
            if meta["parity"][rid] == meta["parity"][nbr]:
                return False

    seed_region_pos = meta["ordered"].index(meta["seed_region"])
    if seed_region_pos % 2 != desired_parity:
        return False

    alt = np.empty_like(out)
    for rid in range(1, len(meta["regions"]) + 1):
        alt[meta["labels"] == rid] = YELLOW if meta["parity"][rid] == 0 else GREY
    alt[meta["wall"]] = GREEN
    alt[meta["start"]] = BLUE
    if np.array_equal(out, alt):
        return False

    return True


def _gen_attempt(counter: int) -> np.ndarray | None:
    branch = counter % 4
    desired_parity = counter % 2
    prefer_end = (counter // 2) % 2 == 0
    h, w, start_cols = _choose_dimensions(branch)
    if not start_cols:
        return None

    start_c = random.choice(list(start_cols))
    path = _trace_path((h, w), (h - 1, start_c))
    wall = np.zeros((h, w), dtype=bool)
    for r, c in path:
        wall[r, c] = True

    labels, regions = _label_regions(~wall)
    if len(regions) < 4:
        return None

    graph = _build_region_graph(labels, path, len(regions))
    ordered = _ordered_path_regions(graph)
    if branch == 2 and len(ordered) < 5:
        return None

    region_id = _select_region_for_seed(ordered, regions, desired_parity, prefer_end)
    if region_id is None:
        return None

    seed = _pick_seed_cell(labels, wall, region_id, start_c)
    if seed is None or wall[seed]:
        return None

    grid = np.zeros((h, w), dtype=int)
    grid[h - 1, start_c] = BLUE
    grid[seed] = RED

    if not _validate_generated(grid, desired_parity):
        return None

    output_signature = hashlib.md5(main(grid).tobytes() + str(grid.shape).encode()).hexdigest()
    if output_signature in _USED_OUTPUT_SIGNATURES:
        return None
    _USED_OUTPUT_SIGNATURES.add(output_signature)
    return grid


def generate_input() -> np.ndarray:
    global _GEN_COUNTER
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt(_GEN_COUNTER)
        if grid is not None:
            _GEN_COUNTER += 1
            return grid
    raise RuntimeError("failed to generate a valid bounce-region-coloring input")
