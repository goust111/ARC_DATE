import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MARKER_COLORS = [1, 2, 3, 4, 6, 7, 8]
GENERATION_ATTEMPTS = 400


def _neighbors(r: int, c: int, h: int, w: int):
    for dr, dc in DIRS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    comp_id = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            comp_id += 1
            q = deque([(r, c)])
            labels[r, c] = comp_id
            while q:
                cr, cc = q.popleft()
                for nr, nc in _neighbors(cr, cc, h, w):
                    if mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = comp_id
                        q.append((nr, nc))
    return labels, comp_id


def _maze_mask(grid: np.ndarray) -> np.ndarray:
    walkable = grid != GREY
    labels, count = _label_components(walkable)
    if count == 0:
        return np.zeros_like(grid, dtype=bool)

    h, w = grid.shape
    border_ids = set(labels[0, :]) | set(labels[h - 1, :]) | set(labels[:, 0]) | set(labels[:, w - 1])
    border_ids.discard(0)

    marker_mask = (grid != BLACK) & (grid != GREY)
    marker_ids = {int(labels[r, c]) for r, c in np.argwhere(marker_mask) if int(labels[r, c]) != 0}
    inside_marker_ids = marker_ids - border_ids
    if inside_marker_ids:
        assert len(inside_marker_ids) == 1, "expected a single enclosed maze component"
        comp_id = next(iter(inside_marker_ids))
        return labels == comp_id

    inside_ids = set(range(1, count + 1)) - border_ids
    if not inside_ids:
        return np.zeros_like(grid, dtype=bool)
    sizes = {comp_id: int(np.count_nonzero(labels == comp_id)) for comp_id in inside_ids}
    comp_id = max(sizes, key=sizes.get)
    return labels == comp_id


def _bfs_dist(mask: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    h, w = mask.shape
    dist = np.full((h, w), -1, dtype=int)
    q = deque([start])
    dist[start] = 0
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors(r, c, h, w):
            if mask[nr, nc] and dist[nr, nc] == -1:
                dist[nr, nc] = dist[r, c] + 1
                q.append((nr, nc))
    return dist


def _territories(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    maze = _maze_mask(grid)
    markers = [(tuple(map(int, rc)), int(grid[tuple(rc)])) for rc in np.argwhere(maze & (grid != BLACK) & (grid != GREY))]
    assert markers, "expected at least one marker"

    dists = [_bfs_dist(maze, pos) for pos, _ in markers]
    stage1 = grid.copy()
    stage1[maze] = BLACK

    for r, c in np.argwhere(maze):
        best_color = None
        best_dist = None
        tie = False
        for idx, dist in enumerate(dists):
            cur = int(dist[r, c])
            if cur < 0:
                continue
            if best_dist is None or cur < best_dist:
                best_dist = cur
                best_color = markers[idx][1]
                tie = False
            elif cur == best_dist:
                tie = True
        if best_color is not None and not tie:
            stage1[r, c] = best_color
    return maze, stage1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Diffuse leaf markers through an enclosed tree corridor, then recolor territory borders maroon.
    """
    g = np.asarray(grid, dtype=int)
    maze, stage1 = _territories(g)
    out = stage1.copy()
    boundary = np.zeros_like(g, dtype=bool)

    h, w = g.shape
    for r in range(h):
        for c in range(w):
            if not maze[r, c]:
                continue
            here = int(stage1[r, c])
            if here in (BLACK, GREY):
                continue
            for nr, nc in ((r + 1, c), (r, c + 1)):
                if not (0 <= nr < h and 0 <= nc < w and maze[nr, nc]):
                    continue
                there = int(stage1[nr, nc])
                if there in (BLACK, GREY) or there == here:
                    continue
                boundary[r, c] = True
                boundary[nr, nc] = True

    out[boundary] = MAROON
    return out


def _tree_neighbors(cells: set[tuple[int, int]], cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    out = []
    for dr, dc in DIRS4:
        nxt = (r + dr, c + dc)
        if nxt in cells:
            out.append(nxt)
    return out


def _grow_tree(h: int, w: int, size: int, branchy: bool) -> set[tuple[int, int]] | None:
    start = (random.randrange(h), random.randrange(w))
    cells = {start}
    while len(cells) < size:
        frontier: list[tuple[int, int]] = []
        weights: list[int] = []
        for r in range(h):
            for c in range(w):
                cell = (r, c)
                if cell in cells:
                    continue
                neighbors = _tree_neighbors(cells, cell)
                if len(neighbors) != 1:
                    continue
                frontier.append(cell)
                parent_degree = len(_tree_neighbors(cells, neighbors[0]))
                if branchy:
                    weights.append(1 + parent_degree)
                else:
                    weights.append(4 - min(parent_degree, 3))
        if not frontier:
            return None
        pick = random.choices(frontier, weights=weights, k=1)[0]
        cells.add(pick)
    return cells


def _tree_degrees(cells: set[tuple[int, int]]) -> dict[tuple[int, int], int]:
    return {cell: len(_tree_neighbors(cells, cell)) for cell in cells}


def _distances_from(cells: set[tuple[int, int]], start: tuple[int, int]) -> dict[tuple[int, int], int]:
    dist = {start: 0}
    q = deque([start])
    while q:
        cell = q.popleft()
        for nxt in _tree_neighbors(cells, cell):
            if nxt not in dist:
                dist[nxt] = dist[cell] + 1
                q.append(nxt)
    return dist


def _choose_marked_leaves(
    cells: set[tuple[int, int]],
    leaves: list[tuple[int, int]],
    mode: str,
) -> list[tuple[int, int]] | None:
    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for i in range(len(leaves)):
        dist = _distances_from(cells, leaves[i])
        for j in range(i + 1, len(leaves)):
            if dist[leaves[j]] % 2 == 0:
                pairs.append((leaves[i], leaves[j]))

    if mode in {"tie", "frontier"}:
        target_even = mode == "tie"
        candidates = []
        for i in range(len(leaves)):
            dist = _distances_from(cells, leaves[i])
            for j in range(i + 1, len(leaves)):
                if (dist[leaves[j]] % 2 == 0) == target_even:
                    candidates.append((leaves[i], leaves[j]))
        if not candidates:
            return None
        return list(random.choice(candidates))

    if len(leaves) < 3:
        return None
    count = random.randint(3, min(5, len(leaves)))
    return random.sample(leaves, count)


def _embed_tree(
    cells: set[tuple[int, int]],
    marked: list[tuple[int, int]],
    tree_h: int,
    tree_w: int,
) -> np.ndarray:
    margin_top = random.randint(1, 4)
    margin_bottom = random.randint(1, 4)
    margin_left = random.randint(1, 4)
    margin_right = random.randint(1, 4)
    h = tree_h + 2 + margin_top + margin_bottom
    w = tree_w + 2 + margin_left + margin_right
    grid = np.full((h, w), BLACK, dtype=int)

    top = margin_top
    left = margin_left
    grid[top : top + tree_h + 2, left : left + tree_w + 2] = GREY
    for r, c in cells:
        grid[top + 1 + r, left + 1 + c] = BLACK

    palette = random.sample(MARKER_COLORS, len(marked))
    for color, (r, c) in zip(palette, marked):
        grid[top + 1 + r, left + 1 + c] = color

    return grid


def _count_tree_edges(cells: set[tuple[int, int]]) -> int:
    total = 0
    for r, c in cells:
        for dr, dc in ((1, 0), (0, 1)):
            if (r + dr, c + dc) in cells:
                total += 1
    return total


def _valid_tree(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    edge_count = _count_tree_edges(cells)
    if edge_count != len(cells) - 1:
        return False
    start = next(iter(cells))
    seen = {start}
    q = deque([start])
    while q:
        cell = q.popleft()
        for nxt in _tree_neighbors(cells, cell):
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return len(seen) == len(cells)


def _gen_attempt() -> np.ndarray | None:
    tree_h = random.randint(6, 11)
    tree_w = random.randint(6, 11)
    max_cells = tree_h * tree_w
    min_size = max(8, (tree_h * tree_w) // 5)
    max_size = max(min_size, (tree_h * tree_w) // 2)
    size = random.randint(min_size, max_size)
    mode = random.choice(["tie", "frontier", "multi"])
    branchy = random.random() < 0.5

    cells = _grow_tree(tree_h, tree_w, size, branchy)
    if cells is None or not _valid_tree(cells):
        return None

    degrees = _tree_degrees(cells)
    leaves = [cell for cell, deg in degrees.items() if deg == 1]
    if len(leaves) < 2:
        return None
    if mode == "multi" and len(leaves) < 3:
        return None
    if random.random() < 0.5 and len(leaves) <= 2:
        return None

    marked = _choose_marked_leaves(cells, leaves, mode)
    if marked is None:
        return None

    grid = _embed_tree(cells, marked, tree_h, tree_w)
    maze = _maze_mask(grid)
    if int(np.count_nonzero(maze)) != len(cells):
        return None

    marker_positions = np.argwhere((grid != BLACK) & (grid != GREY))
    if marker_positions.shape[0] != len(marked):
        return None

    # Verify that every marker sits on a dead end inside the discovered maze.
    for r, c in marker_positions:
        deg = sum(1 for nr, nc in _neighbors(int(r), int(c), *grid.shape) if maze[nr, nc])
        if deg != 1:
            return None

    out = main(grid)
    maroon_count = int(np.count_nonzero(out == MAROON))
    unresolved_count = int(np.count_nonzero(maze & (out == BLACK)))

    if mode == "tie" and unresolved_count == 0:
        return None
    if mode == "frontier" and maroon_count < 2:
        return None
    if mode == "multi" and maroon_count == 0:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid tree-maze input")
