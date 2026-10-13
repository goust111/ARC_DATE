from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
TRACK = 5
WHITE = 8
SEED_POOL = [1, 2, 3, 4, 6, 7, 9]

GEN_ATTEMPTS = 400
SEED_ATTEMPTS = 500
PATH_BUILD_ATTEMPTS = 200
TREE_BUILD_ATTEMPTS = 200

COARSE_MIN = 4
COARSE_MAX = 13
MAX_COARSE_NODES = 20


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _majority_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[counts.argmax()])


def _track_graph(track_mask: np.ndarray) -> dict[tuple[int, int], list[tuple[int, int]]]:
    graph: dict[tuple[int, int], list[tuple[int, int]]] = {}
    h, w = track_mask.shape
    for r, c in np.argwhere(track_mask):
        rr = int(r)
        cc = int(c)
        nbrs = []
        for nr, nc in _neighbors4(rr, cc):
            if 0 <= nr < h and 0 <= nc < w and bool(track_mask[nr, nc]):
                nbrs.append((nr, nc))
        graph[(rr, cc)] = nbrs
    return graph


def _all_pairs_track_distances(
    graph: dict[tuple[int, int], list[tuple[int, int]]]
) -> dict[tuple[int, int], dict[tuple[int, int], int]]:
    dists: dict[tuple[int, int], dict[tuple[int, int], int]] = {}
    for start in graph:
        q = deque([start])
        seen = {start: 0}
        while q:
            cell = q.popleft()
            for nbr in graph[cell]:
                if nbr not in seen:
                    seen[nbr] = seen[cell] + 1
                    q.append(nbr)
        dists[start] = seen
    return dists


def main(grid: np.ndarray) -> np.ndarray:
    """
    Spread seed colors synchronously along a thin track and mark equal-distance front collisions in white.

    Input structure:
    - Black background surrounds a single connected 4-neighbor track.
    - Most track cells are grey; a few single-pixel colored seeds sit directly on the track.
    - The track is one pixel wide and may bend or branch, but all non-background cells belong to that same network.

    Rule:
    - Infer the grey track color as the most frequent non-background color, and treat every non-background cell as passable track.
    - Start simultaneous 4-neighbor growth from every seed and propagate only through track cells.
    - A track cell reached first by exactly one seed color becomes that color.
    - A track cell reached at the same minimum step count by different seed colors becomes white.
    - Seed pixels keep their original colors, and the background stays black.

    Key insight:
    The comparison is graph distance along the winding track, not straight-line distance in the image; white cells are exactly the equal-time Voronoi boundary on that track graph.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _majority_color(g)

    non_bg = [int(v) for v in g.flat if int(v) != bg]
    assert non_bg, "grid must contain a track"
    counts = Counter(non_bg)
    track_color, _ = counts.most_common(1)[0]

    track_mask = g != bg
    seed_cells = [
        (int(r), int(c), int(g[int(r), int(c)]))
        for r, c in np.argwhere((g != bg) & (g != track_color))
    ]
    assert seed_cells, "grid must contain at least one colored seed"

    dist = np.full((h, w), -1, dtype=int)
    owners = [[set() for _ in range(w)] for _ in range(h)]
    q: deque[tuple[int, int, int]] = deque()

    for r, c, color in seed_cells:
        if dist[r, c] == -1:
            dist[r, c] = 0
        if color not in owners[r][c]:
            owners[r][c].add(color)
            q.append((r, c, color))

    while q:
        r, c, color = q.popleft()
        step = int(dist[r, c])
        for nr, nc in _neighbors4(r, c):
            if not (0 <= nr < h and 0 <= nc < w) or not bool(track_mask[nr, nc]):
                continue
            next_step = step + 1
            if dist[nr, nc] == -1 or next_step < dist[nr, nc]:
                dist[nr, nc] = next_step
                owners[nr][nc] = {color}
                q.append((nr, nc, color))
            elif next_step == dist[nr, nc] and color not in owners[nr][nc]:
                owners[nr][nc].add(color)
                q.append((nr, nc, color))

    out = np.full_like(g, bg)
    for r in range(h):
        for c in range(w):
            if not bool(track_mask[r, c]):
                continue
            if len(owners[r][c]) == 1:
                out[r, c] = next(iter(owners[r][c]))
            elif len(owners[r][c]) >= 2:
                out[r, c] = WHITE
            else:
                out[r, c] = g[r, c]

    return out


def _coarse_neighbors(
    cell: tuple[int, int], coarse_h: int, coarse_w: int
) -> list[tuple[int, int]]:
    r, c = cell
    nbrs = []
    for nr, nc in _neighbors4(r, c):
        if 0 <= nr < coarse_h and 0 <= nc < coarse_w:
            nbrs.append((nr, nc))
    return nbrs


def _build_coarse_path(
    coarse_h: int, coarse_w: int, target_nodes: int
) -> tuple[set[tuple[int, int]], set[tuple[tuple[int, int], tuple[int, int]]]] | None:
    min_nodes = max(4, min(target_nodes, 4))
    for _ in range(PATH_BUILD_ATTEMPTS):
        start = (random.randrange(coarse_h), random.randrange(coarse_w))
        path = [start]
        used = {start}

        while len(path) < target_nodes:
            options = []
            end_indices = [0] if len(path) == 1 else [0, -1]
            for idx in end_indices:
                for nbr in _coarse_neighbors(path[idx], coarse_h, coarse_w):
                    if nbr not in used:
                        options.append((idx, nbr))
            if not options:
                break
            idx, nbr = random.choice(options)
            if idx == 0:
                path.insert(0, nbr)
            else:
                path.append(nbr)
            used.add(nbr)

        if len(path) < min_nodes:
            continue
        edges = {
            tuple(sorted((path[i], path[i + 1])))
            for i in range(len(path) - 1)
        }
        return set(path), edges
    return None


def _build_coarse_tree(
    coarse_h: int, coarse_w: int, target_nodes: int
) -> tuple[set[tuple[int, int]], set[tuple[tuple[int, int], tuple[int, int]]]] | None:
    min_nodes = max(5, min(target_nodes, 5))
    for _ in range(TREE_BUILD_ATTEMPTS):
        start = (random.randrange(coarse_h), random.randrange(coarse_w))
        nodes = {start}
        edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
        degree = Counter()

        while len(nodes) < target_nodes:
            options = []
            for node in nodes:
                if degree[node] >= 3:
                    continue
                for nbr in _coarse_neighbors(node, coarse_h, coarse_w):
                    if nbr not in nodes:
                        options.append((node, nbr))
            if not options:
                break
            parent, child = random.choice(options)
            nodes.add(child)
            edge = tuple(sorted((parent, child)))
            edges.add(edge)
            degree[parent] += 1
            degree[child] += 1

        if len(nodes) < min_nodes:
            continue
        if not any(degree[node] >= 3 for node in nodes):
            continue
        return nodes, edges
    return None


def _coarse_to_full_track(
    coarse_h: int,
    coarse_w: int,
    nodes: set[tuple[int, int]],
    edges: set[tuple[tuple[int, int], tuple[int, int]]],
) -> tuple[np.ndarray, dict[tuple[int, int], list[tuple[int, int]]]]:
    h = 2 * coarse_h + 1
    w = 2 * coarse_w + 1
    track_mask = np.zeros((h, w), dtype=bool)

    for r, c in nodes:
        track_mask[2 * r + 1, 2 * c + 1] = True
    for a, b in edges:
        ar, ac = a
        br, bc = b
        fr, fc = 2 * ar + 1, 2 * ac + 1
        tr, tc = 2 * br + 1, 2 * bc + 1
        track_mask[(fr + tr) // 2, (fc + tc) // 2] = True

    return track_mask, _track_graph(track_mask)


def _seed_candidates(
    graph: dict[tuple[int, int], list[tuple[int, int]]]
) -> list[tuple[int, int]]:
    candidates = [cell for cell, nbrs in graph.items() if len(nbrs) <= 2]
    if len(candidates) >= 2:
        return candidates
    return list(graph)


def _seed_layout_ok(
    cells: list[tuple[int, int]],
    dists: dict[tuple[int, int], dict[tuple[int, int], int]],
    min_gap: int,
) -> bool:
    for i, a in enumerate(cells):
        for b in cells[i + 1 :]:
            if dists[a][b] < min_gap:
                return False
    return True


def _seeded_grid(
    track_mask: np.ndarray, seed_cells: list[tuple[int, int]], seed_colors: list[int]
) -> np.ndarray:
    grid = np.full(track_mask.shape, BG, dtype=int)
    grid[track_mask] = TRACK
    for (r, c), color in zip(seed_cells, seed_colors, strict=False):
        grid[r, c] = int(color)
    return grid


def _output_branch_ok(out: np.ndarray, seed_colors: list[int], want_white: bool) -> bool:
    has_white = bool(np.any(out == WHITE))
    if has_white != want_white:
        return False
    color_counts = Counter(int(v) for v in out.flat if int(v) in seed_colors)
    return all(color_counts[color] >= 2 for color in seed_colors)


def _place_seeds(
    track_mask: np.ndarray,
    graph: dict[tuple[int, int], list[tuple[int, int]]],
    want_white: bool,
) -> np.ndarray | None:
    candidates = _seed_candidates(graph)
    if len(candidates) < 2:
        return None

    dists = _all_pairs_track_distances(graph)
    min_gap = 4 if len(graph) >= 12 else 3

    for _ in range(SEED_ATTEMPTS):
        seed_count = random.choice([2, 2, 3, 4])
        seed_count = min(seed_count, len(candidates), len(SEED_POOL))
        if seed_count < 2:
            continue

        shuffled = candidates[:]
        random.shuffle(shuffled)
        picked: list[tuple[int, int]] = []
        for cell in shuffled:
            if all(dists[cell][other] >= min_gap for other in picked):
                picked.append(cell)
                if len(picked) == seed_count:
                    break
        if len(picked) < seed_count or not _seed_layout_ok(picked, dists, min_gap):
            continue

        colors = random.sample(SEED_POOL, seed_count)
        grid = _seeded_grid(track_mask, picked, colors)
        out = main(grid)
        if _output_branch_ok(out, colors, want_white):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        want_white = random.choice([True, False])
        mode = random.choice(["path", "branchy"])

        coarse_h = random.randint(COARSE_MIN, COARSE_MAX)
        coarse_w = random.randint(COARSE_MIN, COARSE_MAX)
        max_nodes = min(MAX_COARSE_NODES, coarse_h * coarse_w)
        if mode == "path":
            target_nodes = random.randint(4, max_nodes)
            built = _build_coarse_path(coarse_h, coarse_w, target_nodes)
        else:
            target_nodes = random.randint(6, max_nodes)
            built = _build_coarse_tree(coarse_h, coarse_w, target_nodes)
        if built is None:
            continue

        nodes, edges = built
        track_mask, graph = _coarse_to_full_track(coarse_h, coarse_w, nodes, edges)
        if max(track_mask.shape) > 30 or len(graph) < 6:
            continue

        grid = _place_seeds(track_mask, graph, want_white)
        if grid is None:
            continue

        non_bg_colors = Counter(int(v) for v in grid.flat if int(v) != BG)
        if non_bg_colors[TRACK] <= 1:
            continue
        if non_bg_colors[TRACK] != max(non_bg_colors.values()):
            continue
        return grid

    raise RuntimeError("failed to generate a valid ARC sample")
