from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5

GRID_MIN = 9
GRID_MAX = 18
TREE_SIZE_MIN = 10
GEN_ATTEMPTS = 400

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _is_edge(shape: tuple[int, int], r: int, c: int) -> bool:
    return r == 0 or c == 0 or r == shape[0] - 1 or c == shape[1] - 1


def _color_positions(g: np.ndarray, color: int) -> list[tuple[int, int]]:
    return [tuple(map(int, p)) for p in np.argwhere(g == color)]


def _assert_connected_tree(mask: np.ndarray) -> set[tuple[int, int]]:
    cells = {tuple(map(int, p)) for p in np.argwhere(mask)}
    assert cells, "expected at least one channel cell"

    start = next(iter(cells))
    seen = {start}
    q = deque([start])
    edge_count = 0
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cells:
                edge_count += 1
                if (nr, nc) not in seen:
                    seen.add((nr, nc))
                    q.append((nr, nc))

    assert seen == cells, "black channel must be 4-connected"
    assert edge_count // 2 == len(cells) - 1, "black channel must be acyclic"
    return cells


def _unique_entry_neighbor(g: np.ndarray, marker: tuple[int, int]) -> tuple[int, int]:
    neighbors = []
    for nr, nc in _neighbors4(*marker):
        if _in_bounds(g.shape, nr, nc) and int(g[nr, nc]) == BLACK:
            neighbors.append((nr, nc))
    assert len(neighbors) == 1, "each edge marker must touch exactly one channel cell"
    return neighbors[0]


def _path_on_tree(
    cells: set[tuple[int, int]], start: tuple[int, int], end: tuple[int, int]
) -> list[tuple[int, int]]:
    q = deque([start])
    parent = {start: None}
    while q and end not in parent:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            nxt = (nr, nc)
            if nxt in cells and nxt not in parent:
                parent[nxt] = (r, c)
                q.append(nxt)

    assert end in parent, "tree path must connect start and end"
    path = []
    cur = end
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow the unique path in a tree-shaped black channel network between a blue
    edge marker and a red edge marker, then recolor the traversed black cells blue.
    """
    g = np.asarray(grid, dtype=int)

    blue_pixels = _color_positions(g, BLUE)
    red_pixels = _color_positions(g, RED)
    assert len(blue_pixels) == 1 and len(red_pixels) == 1, "expected one blue and one red marker"

    blue = blue_pixels[0]
    red = red_pixels[0]
    assert _is_edge(g.shape, *blue) and _is_edge(g.shape, *red), "markers must sit on the grid edge"

    channel_cells = _assert_connected_tree(g == BLACK)
    start = _unique_entry_neighbor(g, blue)
    end = _unique_entry_neighbor(g, red)

    path = _path_on_tree(channel_cells, start, end)
    out = g.copy()
    for r, c in path:
        out[r, c] = BLUE
    out[blue] = BLUE
    out[red] = RED
    return out


def _interior_neighbors(shape: tuple[int, int], cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    out = []
    for nr, nc in _neighbors4(r, c):
        if 1 <= nr < shape[0] - 1 and 1 <= nc < shape[1] - 1:
            out.append((nr, nc))
    return out


def _edge_attachments(shape: tuple[int, int], tree: set[tuple[int, int]]):
    attachments = []
    for r, c in tree:
        if r == 1:
            attachments.append(((0, c), (r, c), "top"))
        if r == shape[0] - 2:
            attachments.append(((shape[0] - 1, c), (r, c), "bottom"))
        if c == 1:
            attachments.append(((r, 0), (r, c), "left"))
        if c == shape[1] - 2:
            attachments.append(((r, shape[1] - 1), (r, c), "right"))
    return attachments


def _tree_distance(tree: set[tuple[int, int]], start: tuple[int, int], end: tuple[int, int]) -> int:
    return len(_path_on_tree(tree, start, end))


def _choose_markers(shape: tuple[int, int], tree: set[tuple[int, int]]):
    attachments = _edge_attachments(shape, tree)
    if len(attachments) < 2:
        return None

    random.shuffle(attachments)
    best = None
    for i in range(len(attachments)):
        for j in range(i + 1, len(attachments)):
            edge_a, inner_a, _side_a = attachments[i]
            edge_b, inner_b, _side_b = attachments[j]
            if edge_a == edge_b or inner_a == inner_b:
                continue
            dist = _tree_distance(tree, inner_a, inner_b)
            if dist < 4:
                continue
            score = (dist, edge_a != edge_b)
            if best is None or score > best[0]:
                best = (score, attachments[i], attachments[j])

    if best is None:
        return None
    return best[1], best[2]


def _transform_grid(g: np.ndarray) -> np.ndarray:
    out = np.rot90(g, random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    return out.copy()


def _sample_anchor(shape: tuple[int, int], side: str) -> tuple[tuple[int, int], tuple[int, int]]:
    h, w = shape
    if side == "top":
        c = random.randint(1, w - 2)
        return (0, c), (1, c)
    if side == "bottom":
        c = random.randint(1, w - 2)
        return (h - 1, c), (h - 2, c)
    if side == "left":
        r = random.randint(1, h - 2)
        return (r, 0), (r, 1)
    r = random.randint(1, h - 2)
    return (r, w - 1), (r, w - 2)


def _random_main_path(
    shape: tuple[int, int], start: tuple[int, int], end: tuple[int, int]
) -> list[tuple[int, int]] | None:
    current = start
    path = [start]
    seen = {start}

    while current != end:
        candidates = [p for p in _interior_neighbors(shape, current) if p not in seen]
        if not candidates:
            return None

        random.shuffle(candidates)
        candidates.sort(key=lambda p: (abs(p[0] - end[0]) + abs(p[1] - end[1]) + random.randint(0, 3)))

        next_cell = None
        for cand in candidates:
            cur_dist = abs(current[0] - end[0]) + abs(current[1] - end[1])
            new_dist = abs(cand[0] - end[0]) + abs(cand[1] - end[1])
            if new_dist <= cur_dist or random.random() < 0.2:
                next_cell = cand
                break
        if next_cell is None:
            next_cell = candidates[0]

        path.append(next_cell)
        seen.add(next_cell)
        current = next_cell

        if len(path) > (shape[0] - 2) * (shape[1] - 2):
            return None

    return path


def _build_tree(shape: tuple[int, int], target_size: int):
    sides = ["top", "bottom", "left", "right"]
    side_a = random.choice(sides)
    side_b = random.choice([s for s in sides if s != side_a or random.random() < 0.5])

    edge_a, anchor_a = _sample_anchor(shape, side_a)
    edge_b, anchor_b = _sample_anchor(shape, side_b)
    if anchor_a == anchor_b:
        return None
    if abs(anchor_a[0] - anchor_b[0]) + abs(anchor_a[1] - anchor_b[1]) < 5:
        return None

    path = _random_main_path(shape, anchor_a, anchor_b)
    if path is None or len(path) < 5:
        return None

    tree = set(path)
    branch_bases = list(tree)
    while len(tree) < target_size and branch_bases:
        base = random.choice(branch_bases)
        length = random.randint(1, 5)
        current = base
        grown = []

        for _ in range(length):
            options = [p for p in _interior_neighbors(shape, current) if p not in tree]
            if not options:
                break
            random.shuffle(options)
            nxt = options[0]
            tree.add(nxt)
            grown.append(nxt)
            current = nxt
            if len(tree) >= target_size:
                break

        if not grown:
            branch_bases.remove(base)
        else:
            branch_bases.extend(grown)

    return tree, (edge_a, anchor_a), (edge_b, anchor_b)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    max_tree = max(TREE_SIZE_MIN, min((h - 2) * (w - 2) // 2, 45))
    if max_tree < TREE_SIZE_MIN:
        return None

    target_size = random.randint(TREE_SIZE_MIN, max_tree)
    built = _build_tree((h, w), target_size)
    if built is None:
        return None
    tree, (blue_edge, blue_inner), (red_edge, red_inner) = built
    if len(tree) < TREE_SIZE_MIN or _tree_distance(tree, blue_inner, red_inner) < 4:
        return None

    grid = np.full((h, w), GREY, dtype=int)
    for r, c in tree:
        grid[r, c] = BLACK
    grid[blue_edge] = BLUE
    grid[red_edge] = RED

    transformed = _transform_grid(grid)

    blue_pixels = _color_positions(transformed, BLUE)
    red_pixels = _color_positions(transformed, RED)
    if len(blue_pixels) != 1 or len(red_pixels) != 1:
        return None

    try:
        out = main(transformed)
    except AssertionError:
        return None

    path_blue = int(np.sum((transformed != BLUE) & (out == BLUE)))
    if path_blue < 4:
        return None
    return transformed


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid tree-maze ARC input")
